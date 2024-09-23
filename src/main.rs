use clap::Parser;
use indicatif::ProgressIterator;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams, Token};
use llama_cpp_sys::llama_token_data;
use machine_info::Machine;
use std::{
    fmt::Write,
    io::Read,
    sync::{Arc, Mutex},
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

fn init_tracing() {
    let format = tracing_subscriber::fmt::layer().compact();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(
        tracing_subscriber::EnvFilter::default()
            .add_directive(tracing_subscriber::filter::LevelFilter::WARN.into()),
    );

    tracing_subscriber::registry()
        .with(format)
        .with(filter)
        .init();
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to the .gguf model file
    #[arg(short, long)]
    model: String,
}
pub struct PeekSampler {
    eos: Token,
    candidates: Arc<Mutex<Vec<llama_token_data>>>,
}

impl llama_cpp::Sampler for PeekSampler {
    fn sample(
        &mut self,
        context: *mut llama_cpp_sys::llama_context,
        _tokens: &[Token],
        mut candidates_p: llama_cpp_sys::llama_token_data_array,
    ) -> Token {
        unsafe {
            llama_cpp_sys::llama_sample_top_p(
                context,
                std::ptr::addr_of_mut!(candidates_p),
                0.9995,
                5,
            );
        }

        for i in 0..candidates_p.size {
            self.candidates
                .lock()
                .unwrap()
                .push(unsafe { *candidates_p.data.wrapping_add(i) });
        }
        return self.eos; // Don't produce any tokens! (This sampler doesn't sample.)
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct Strip {
    leadup: String,
    punchline: String,
}

fn get_strips(path: &str) -> Vec<Strip> {
    let file = std::fs::File::open(path).unwrap();
    let mut reader = csv::Reader::from_reader(file);

    let mut res = vec![];
    for result in reader.records() {
        let record = result.unwrap();

        let (prefix_lines, last_line) = record.get(3).unwrap().trim().rsplit_once("\n").unwrap();

        let leadup: String;
        let ending: String;
        if last_line.len() > 50 {
            leadup = prefix_lines.to_owned();
            ending = last_line.to_string();
        } else {
            let (leadup_first, penultimate) = prefix_lines.rsplit_once("\n").unwrap();
            leadup = leadup_first.to_owned();
            ending = format!("{penultimate}\n{last_line}");
        }

        if let Some((speaker, punchline)) = ending.split_once(": ") {
            if let Some((punchline_first_word, punchline_rest)) =
                punchline.split_once(char::is_whitespace)
            {
                let strip = Strip {
                    leadup: format!("{}\n{}: {}", leadup, speaker, punchline_first_word),
                    punchline: punchline_rest.to_owned(),
                };

                res.push(strip);
            }
        }
    }
    return res;
}

fn megabytes(bytes: usize) -> usize {
    return bytes >> 20;
}

#[derive(Default)]
struct Stats {
    details: String,
    tok_times: Vec<std::time::Duration>,
    aheads: Vec<u32>,
    prob_aheads: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

//fn predict_strip(strip: &Strip, ctx: &mut LlamaSession) {
fn predict_strip(strip: &Strip, model: &LlamaModel, stats: &mut Stats) {
    let mut params = SessionParams::default();
    params.n_ctx += 1000;

    let mut ctx = model
        .create_session(params)
        .expect("Failed to create session");

    ctx.set_context(&strip.leadup).unwrap();

    {
        let before = std::time::Instant::now();
        ctx.deep_copy().unwrap();
        let copy_time = before.elapsed();
        writeln!(
            stats.details,
            "Context copy time: {} microseconds or {:.6} seconds",
            copy_time.as_micros(),
            copy_time.as_secs_f32()
        )
        .unwrap();
    }

    let punch_toks = ctx
        .model()
        .tokenize_bytes(&strip.punchline, false, false)
        .unwrap();

    write!(
        stats.details,
        "...{}\n",
        strip
            .leadup
            .chars()
            .skip(strip.leadup.len() - 150)
            .collect::<String>()
    )
    .unwrap();

    let mut tok_s = "   ".to_string();
    let mut ahead_s = "   ".to_string();
    let mut prob_ahead_s = "   ".to_string();
    let mut logit_s = "   ".to_string();
    let mut prob_s = "   ".to_string();

    let mut optimistic_cost = 1.0;

    let mut punchline: String = String::new();
    for (tok_i, tok) in punch_toks.iter().enumerate() {
        let candidates = Arc::new(Mutex::new(vec![]));
        let peek_sampler = PeekSampler {
            eos: ctx.model().eos(),
            candidates: candidates.clone(),
        };

        let before = std::time::Instant::now();
        let mut completion_res = ctx
            .start_completing_with(peek_sampler, 1)
            .expect("Completion error!");
        let _ = completion_res.next_token();
        stats.tok_times.push(before.elapsed());

        let candidates: &Vec<llama_token_data> = &candidates.lock().unwrap();

        let mut ahead = 0;
        let mut prob_ahead = 0.0;
        let mut found_cand = None;
        for cand in candidates.iter() {
            if Token(cand.id) == *tok {
                found_cand = Some(cand);
                break;
            }
            ahead += 1;
            prob_ahead += cand.p;
        }

        write!(tok_s, "{:>10}", ctx.model().decode_tokens(&[*tok])).unwrap();

        if let Some(cand) = found_cand {
            write!(ahead_s, "{:>10}", ahead).unwrap();
            write!(prob_ahead_s, "{:>9.2}%", prob_ahead * 100.0).unwrap();
            write!(logit_s, "{:>10.2}", cand.logit).unwrap();
            write!(prob_s, "{:>9.2}%", cand.p * 100.0).unwrap();
            stats.aheads.push(ahead);
            stats.probs.push(cand.p);
            stats.prob_aheads.push(prob_ahead);
            stats.logits.push(cand.logit);
        } else {
            write!(ahead_s, " -------- ").unwrap();
            write!(prob_ahead_s, " -------- ").unwrap();
            write!(logit_s, " -------- ").unwrap();
            write!(prob_s, " -------- ").unwrap();
            stats.aheads.push(10000);
            stats.probs.push(0.0001);
            stats.prob_aheads.push(0.9999);
            stats.logits.push(-9999.9);
        }

        let anagram_restriction = 0.8 - (0.75 * (tok_i as f64 / (punch_toks.len() as f64)));

        optimistic_cost *= (*stats.aheads.last().unwrap() as f64 * anagram_restriction) + 1.0;

        ctx.advance_context_with_tokens(&[*tok])
            .expect("Advancement error!");
        punchline.push_str(&ctx.model().decode_tokens([*tok].iter()));
    }

    write!(
        stats.details,
        "{tok_s}\n{logit_s}\n{prob_s}\n{ahead_s}\n{prob_ahead_s}\noptimistic cost: {:.2e}  average tok time: {:.0}\n",
        optimistic_cost,
        stats.tok_times.iter().map(|d| d.as_micros()).sum::<u128>() as f64 /
             (stats.tok_times.len()) as f64
    )
    .unwrap();
}

fn price_out_strip(
    strip: &Strip,
    model: &LlamaModel,
    aheads_limit: u32,
    prob_aheads_limit: f32,
    logits_limit: f32,
    probs_limit: f32,
) -> (f64, f64, f64, f64) {
    let mut params = SessionParams::default();
    params.n_ctx += 1000;

    let mut ctx = model
        .create_session(params)
        .expect("Failed to create session");

    ctx.set_context(&strip.leadup).unwrap();

    let punch_toks = ctx
        .model()
        .tokenize_bytes(&strip.punchline, false, false)
        .unwrap();

    let mut a_total = 1.0;
    let mut pa_total = 1.0;
    let mut l_total = 1.0;
    let mut p_total = 1.0;
    for (tok_i, tok) in punch_toks.iter().enumerate() {
        let candidates = Arc::new(Mutex::new(vec![]));
        let peek_sampler = PeekSampler {
            eos: ctx.model().eos(),
            candidates: candidates.clone(),
        };

        let mut completion_res = ctx
            .start_completing_with(peek_sampler, 1)
            .expect("Completion error!");
        let _ = completion_res.next_token();

        let candidates: &Vec<llama_token_data> = &candidates.lock().unwrap();

        let mut pa_count = None;
        let mut l_count = None;
        let mut p_count = None;

        let mut i = 0;
        let mut prob_ahead = 0.0;
        for cand in candidates.iter() {
            i += 1;
            prob_ahead += cand.p;

            if pa_count.is_none() && prob_ahead > prob_aheads_limit {
                pa_count = Some(i);
            }
            if l_count.is_none() && cand.logit < logits_limit {
                l_count = Some(i);
            }
            if p_count.is_none() && cand.p < probs_limit {
                p_count = Some(i);
            }
        }
        if pa_count.is_none() {
            pa_count = Some(i);
        }
        if l_count.is_none() {
            l_count = Some(i);
        }
        if p_count.is_none() {
            p_count = Some(i);
        }
        let a_count = std::cmp::min(candidates.len() as u32, aheads_limit);

        let anagram_restriction = 0.8 - (0.75 * (tok_i as f64 / (punch_toks.len() as f64)));

        let a_choices = f64::max(a_count as f64 * anagram_restriction, 1.0);
        let pa_choices = f64::max(pa_count.unwrap() as f64 * anagram_restriction, 1.0);
        let l_choices = f64::max(l_count.unwrap() as f64 * anagram_restriction, 1.0);
        let p_choices = f64::max(p_count.unwrap() as f64 * anagram_restriction, 1.0);

        print!(
            "{}|{}|{:.0}:{:.2}  ",
            candidates.len(),
            pa_count.unwrap(),
            pa_choices,
            anagram_restriction
        );

        a_total *= a_choices;
        pa_total *= pa_choices;
        l_total *= l_choices;
        p_total *= p_choices;

        ctx.advance_context_with_tokens(&[*tok])
            .expect("Advancement error!");
    }

    println!();

    (a_total, pa_total, l_total, p_total)
}

fn log_gpu(machine: &machine_info::Machine) {
    let mut any = false;
    for card in machine.graphics_status() {
        any = true;
        println!(
            "({}: {} MB used)",
            card.id,
            megabytes(card.memory_used as usize)
        );
    }
    if !any {
        println!("(no graphics cards)")
    }
}

fn main() {
    init_tracing();
    let mut peak_vram: u64 = 0;

    let machine = machine_info::Machine::new();
    for card in machine.graphics_status() {
        peak_vram = std::cmp::max(peak_vram, card.memory_used);
    }
    // Create a model from anything that implements `AsRef<Path>`:
    let args = Args::parse();

    let mut model_params = LlamaParams::default();
    model_params.n_gpu_layers = 1000000;
    let model = LlamaModel::load_from_file(args.model, model_params).expect("Could not load model");

    println!(
        "Estimated session size: {} / {}",
        megabytes(
            model
                .estimate_session_size(&SessionParams::default())
                .host_memory
        ),
        megabytes(
            model
                .estimate_session_size(&SessionParams::default())
                .device_memory
        )
    );

    let mut prefix_1663: String = String::default();
    std::fs::File::open("1663-prefix.txt")
        .unwrap()
        .read_to_string(&mut prefix_1663)
        .unwrap();

    println!(
        "prefix tokens: {}",
        model
            .tokenize_bytes(prefix_1663, true, false)
            .unwrap()
            .len()
    );

    // // A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
    // // several gigabytes large, a session is typically a few dozen to a hundred megabytes!
    // let mut ctx = model
    //     .create_session(SessionParams::default())
    //     .expect("Failed to create session");

    let strips = get_strips("/workspace/qwantzle-search/strips.csv");

    let mut toks_of_strips = vec![];
    for strip in &strips {
        toks_of_strips.push(
            model
                .tokenize_bytes(&strip.leadup, true, false)
                .unwrap()
                .len()
                + model
                    .tokenize_bytes(&strip.punchline, true, false)
                    .unwrap()
                    .len(),
        );
    }
    toks_of_strips.sort();

    println!(
        "(Percentile) tokens in a strip: (75) {}  (90) {}  (95) {}  (99) {}  (max) {}",
        toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.75).floor() as usize],
        toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.90).floor() as usize],
        toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.95).floor() as usize],
        toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.99).floor() as usize],
        toks_of_strips.last().unwrap(),
    );

    let representative_strips: Vec<&Strip> = strips
        .iter()
        .filter(|s| s.punchline.len() > 100 && s.punchline.len() < 120)
        .collect();

    let mut stats = Stats::default();
    for strip in representative_strips.iter().take(10).progress() {
        predict_strip(&strip, &model, &mut stats);
        for card in machine.graphics_status() {
            peak_vram = std::cmp::max(peak_vram, card.memory_used);
        }
    }
    std::fs::write("details.txt", stats.details).unwrap();

    let mut stats_csv = csv::Writer::from_path("stats.csv").unwrap();

    stats_csv
        .write_record(&[
            "microseconds",
            "tokens ahead",
            "probability ahead",
            "logit score",
            "probability score",
        ])
        .unwrap();

    for i in 0..stats.tok_times.len() {
        stats_csv
            .serialize((
                stats.tok_times[i].as_micros(),
                stats.aheads[i],
                stats.prob_aheads[i],
                stats.logits[i],
                stats.probs[i],
            ))
            .unwrap();
    }

    stats_csv.flush().unwrap();

    let mut aheads = stats.aheads;
    let mut prob_aheads = stats.prob_aheads;
    let mut logits = stats.logits;
    let mut probs = stats.probs;

    aheads.sort(); // higher is harder
    prob_aheads.sort_by(|a, b| a.partial_cmp(b).unwrap()); // higher is harder
    logits.sort_by(|a, b| a.partial_cmp(b).unwrap());
    logits.reverse(); // lower is harder
    probs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    probs.reverse(); // lower is harder

    for p_limit in [0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.995] {
        let aheads_limit = aheads[((aheads.len() - 1) as f32 * p_limit).floor() as usize];
        let prob_aheads_limit =
            prob_aheads[((prob_aheads.len() - 1) as f32 * p_limit).floor() as usize];
        let logits_limit = logits[((logits.len() - 1) as f32 * p_limit).floor() as usize];
        let probs_limit = probs[((probs.len() - 1) as f32 * p_limit).floor() as usize];

        println!(
            "At {:.3}%, aheads is {}, prob_aheads is {:.4} logits is {:.2}, prob is {:.7}",
            p_limit, aheads_limit, prob_aheads_limit, logits_limit, probs_limit
        );
    }
    println!("Peak VRAM used: {}", megabytes(peak_vram as usize));
}

/*
       let candidates: &Vec<llama_token_data> = &candidates.lock().unwrap();

       println!("{} ...: ({} candidates)", punchline, candidates.len());

       let mut tok_s = "   ".to_string();
       let mut prb_s = "   ".to_string();
       let mut lgt_s = "   ".to_string();

       for candidate in candidates.iter().take(10) {
           let one_tok = vec![Token(candidate.id)];
           write!(tok_s, "{:>10}", ctx.model().decode_tokens(one_tok.iter())).unwrap();
           if candidate.p < 0.01 {
               write!(prb_s, "{:>9.2}%", candidate.p * 100.0).unwrap();
           } else if candidate.p < 0.1 {
               write!(prb_s, "{:>9.1}%", candidate.p * 100.0).unwrap();
           } else {
               write!(prb_s, "{:>9.0}%", candidate.p * 100.0).unwrap();
           }
           write!(lgt_s, "{:>10.1}", candidate.logit).unwrap();
       }
       print!("{}\n{}\n{}\n", tok_s, prb_s, lgt_s);
*/

/*

let p_limit = 0.97;
let aheads_limit = aheads[((aheads.len() - 1) as f32 * p_limit).floor() as usize];
let prob_aheads_limit =
    prob_aheads[((prob_aheads.len() - 1) as f32 * p_limit).floor() as usize];
let logits_limit = logits[((logits.len() - 1) as f32 * p_limit).floor() as usize];
let probs_limit = probs[((probs.len() - 1) as f32 * p_limit).floor() as usize];

println!(
    "At {:.3}%, aheads is {}, prob_aheads is {:.4} logits is {:.2}, prob is {:.7}",
    p_limit, aheads_limit, prob_aheads_limit, logits_limit, probs_limit
);

let mut needed = 20;
for strip in strips.iter() {
    if strip.punchline.len() > 85 && strip.punchline.len() < 115 {
        needed -= 1;
        let (a, pa, l, p) = price_out_strip(
            strip,
            &model,
            aheads_limit,
            prob_aheads_limit,
            logits_limit,
            probs_limit,
        );

        println!(
            "...{} | {}",
            strip
                .leadup
                .chars()
                .skip(strip.leadup.len() - 150)
                .collect::<String>(),
            strip.punchline
        );
        println!(
            "Strip punchline length: {} chars, {} tokens",
            strip.punchline.len(),
            model
                .tokenize_bytes(strip.punchline.to_string(), false, false)
                .unwrap()
                .len()
        );
        println!(
            "Aheads: {:.2e}   Prob aheads: {:.2e}   Logits: {:.2e}   Probs: {:.2e}",
            a, pa, l, p
        );
    }
    if needed <= 0 {
        break;
    }
} */
