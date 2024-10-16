use clap::Parser;
use indicatif::ProgressIterator;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams, Token};
use llama_cpp_sys::llama_token_data;
use std::{
    fmt::Write,
    sync::{Arc, Mutex},
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod llm;
mod pool;
mod search;
mod strip;

use strip::{get_strips, Strip};

fn init_tracing(args: &Args) {
    let format = tracing_subscriber::fmt::layer().compact();
    let level = if args.verbose > 0 {
        tracing_subscriber::filter::LevelFilter::INFO.into()
    } else {
        tracing_subscriber::filter::LevelFilter::ERROR.into()
    };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or(tracing_subscriber::EnvFilter::default().add_directive(level));

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

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Don't store the model on the GPU.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    no_gpu: bool,

    /// Calculate costs and calibrate them against a search
    #[arg(long, action = clap::ArgAction::SetTrue)]
    calibrate_costs: bool,

    /// String to prepend to every strip. Use "l" to add a message about what the longest word in
    /// the punchline is.
    #[arg(short, long, default_value(""))]
    prompt_prefix: String,

    #[arg(short, long, default_value("1"))]
    workers: u8,
}

fn megabytes(bytes: usize) -> usize {
    return bytes >> 20;
}

#[derive(Default)]
struct Stats {
    details: String,
    tok_times: Vec<std::time::Duration>,
    ctx_cpy_times: Vec<std::time::Duration>,
    truncate_times: Vec<std::time::Duration>,
    aheads: Vec<u32>,
    prob_aheads: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    strip_avg_probs: Vec<Vec<f64>>,
}

fn percentile<T: PartialOrd + Copy, U: num_traits::ToPrimitive>(
    samples: &Vec<T>,
    position: U,
    reverse: bool,
) -> T {
    let mut samples_sorted = samples.clone();
    samples_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if reverse {
        samples_sorted.reverse();
    }

    samples_sorted[((samples_sorted.len() - 1) as f32 * (position.to_f32().unwrap() / 100.0) + 0.5)
        .floor() as usize]
}

macro_rules! percentile_table_2_digits {
    ($samples:expr, ($($pos:expr),*), $fmt:literal, $reverse:expr) => {
        {
            let mut res = String::new();
            $(write!(res, "({}%) {}  ", $pos, format!($fmt, percentile($samples, $pos, $reverse)))
                .unwrap();)*
            res
        }
    };
}

macro_rules! percentile_table_2_digits_percentage {
    ($samples:expr, ($($pos:expr),*), $fmt:literal, $reverse:expr) => {
        {
            let new_samples = $samples.iter().map(|v| 100.0 * v).collect::<Vec<_>>();
            let mut res = String::new();
            $(write!(res, "({}%) {}%  ", $pos, format!($fmt, percentile(&new_samples, $pos, $reverse)))
                .unwrap();)*
            res
        }
    };
}

//fn predict_strip(strip: &Strip, ctx: &mut LlamaSession) {
fn predict_strip(strip: &Strip, model: &LlamaModel, stats: &mut Stats) -> (f64, f64) {
    let toks_needed = model
        .tokenize_bytes(&strip.leadup, true, false)
        .unwrap()
        .len()
        + model
            .tokenize_bytes(&strip.punchline, false, false)
            .unwrap()
            .len()
        + 2;
    let mut params = SessionParams::default();
    params.n_ctx = toks_needed as u32;

    let mut ctx = model
        .create_session(params)
        .expect("Failed to create session");

    ctx.set_context(&strip.leadup).unwrap();

    let toks_without_space = model
        .tokenize_bytes(&strip.punchline, false, false)
        .unwrap();

    let toks_with_space = model
        .tokenize_bytes(&format!(" {}", strip.punchline), false, false)
        .unwrap();

    let punch_toks = if toks_with_space.len() <= toks_without_space.len() {
        writeln!(stats.details, "Padding the suffix with a space.").unwrap();
        toks_with_space
    } else {
        writeln!(stats.details, "Not padding the suffix with a space.").unwrap();
        toks_without_space
    };

    // {
    //     // Determine how long copy and truncation take.
    //     let before_cpy = std::time::Instant::now();
    //     let mut copied_ctx = ctx.deep_copy().unwrap();
    //     stats.ctx_cpy_times.push(before_cpy.elapsed());

    //     let before_trn = std::time::Instant::now();
    //     copied_ctx
    //         .truncate_context(ctx.context_size() - 10)
    //         .unwrap();
    //     stats.truncate_times.push(before_trn.elapsed());
    // }

    let mut letter_pool = pool::LetterPool::from_text(&strip.punchline);

    write!(
        stats.details,
        "{}...{}\n",
        strip.leadup.chars().take(100).collect::<String>(),
        strip
            .leadup
            .chars()
            .skip(strip.leadup.len() - 150)
            .collect::<String>()
    )
    .unwrap();

    stats.strip_avg_probs.push(vec![]);
    let mut tok_s = "   ".to_string();
    let mut ahead_s = "   ".to_string();
    let mut ahead_pv_s = "   ".to_string();
    let mut prob_ahead_s = "   ".to_string();
    let mut logit_s = "   ".to_string();
    let mut prob_s = "   ".to_string();

    let mut optimistic_cost = 1.0;
    let mut overall_probability: f64 = 1.0;

    let mut punchline: String = String::new();
    for tok in &punch_toks {
        let candidates = Arc::new(Mutex::new(vec![]));
        let peek_sampler = llm::PeekSampler {
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
        let mut ahead_and_pool_valid = 0;
        let mut prob_ahead = 0.0;
        let mut found_cand = None;
        for cand in candidates.iter() {
            if Token(cand.id) == *tok {
                found_cand = Some(cand);
                break;
            }
            ahead += 1;
            if letter_pool.has(Token(cand.id), model) {
                ahead_and_pool_valid += 1;
            }

            prob_ahead += cand.p;
        }

        write!(tok_s, "{:>10}", ctx.model().decode_tokens(&[*tok])).unwrap();

        if let Some(cand) = found_cand {
            write!(ahead_s, "{:>10}", ahead).unwrap();
            write!(ahead_pv_s, "{:>10}", ahead_and_pool_valid).unwrap();
            write!(prob_ahead_s, "{:>9.2}%", prob_ahead * 100.0).unwrap();
            write!(logit_s, "{:>10.2}", cand.logit).unwrap();
            write!(prob_s, "{:>9.2}%", cand.p * 100.0).unwrap();
            stats.aheads.push(ahead);
            stats.probs.push(cand.p);
            stats.prob_aheads.push(prob_ahead);
            stats.logits.push(cand.logit);
            overall_probability *= cand.p as f64;
        } else {
            write!(ahead_s, " -------- ").unwrap();
            write!(ahead_pv_s, " -------- ").unwrap();
            write!(prob_ahead_s, " -------- ").unwrap();
            write!(logit_s, " -------- ").unwrap();
            write!(prob_s, " -------- ").unwrap();
            stats.aheads.push(10000);
            stats.probs.push(0.0001);
            stats.prob_aheads.push(0.9999);
            stats.logits.push(-9999.9);
            overall_probability *= 0.0001 as f64;
        }

        let toks_so_far = stats.strip_avg_probs.last().unwrap().len() + 1;
        stats
            .strip_avg_probs
            .last_mut()
            .unwrap()
            .push(f64::powf(overall_probability, 1.0 / (toks_so_far as f64)));

        // let anagram_restriction = 0.8 - (0.75 * (tok_i as f64 / (punch_toks.len() as f64)));

        optimistic_cost *= ahead_and_pool_valid as f64 + 1.0;

        ctx.advance_context_with_tokens(&[*tok])
            .expect("Advancement error!");
        punchline.push_str(&model.decode_tokens([*tok].iter()));
        letter_pool.remove(*tok, &model);
    }
    assert!(letter_pool.size() == 0);

    let average_probability = f64::powf(overall_probability, 1.0 / punch_toks.len() as f64);

    write!(
        stats.details,
        "{tok_s}\n{logit_s}\n{prob_s}\n{ahead_s}\n{ahead_pv_s}\n{prob_ahead_s}\noptimistic cost: {:.2e}  average probability: {:.1}%  average tok time: {:.0}\n",
        optimistic_cost,  average_probability * 100.0,
        stats.tok_times.iter().map(|d| d.as_micros()).sum::<u128>() as f64 /
             (stats.tok_times.len()) as f64
    )
    .unwrap();

    return (optimistic_cost, average_probability);
}

fn calibrate_costs(strips: &Vec<Strip>, model: &LlamaModel, args: &Args) {
    let char_min = 40;
    let char_max = 65;

    let excused_strips = [
        1059, // Digits, being one-character tokens, make this one tough (1663 has no digits)
        886,  // "JODIE FOSTER" is in all caps and she's not mentioned or aluded to in the leadup
        2434, // Weird structure; I don't expect an LLM to figure this one out.
    ];

    let small_strips: Vec<&Strip> = strips
        .iter()
        .filter(|s| s.punchline.len() > char_min && s.punchline.len() <= char_max)
        .filter(|s| !s.punchline.contains("\n"))
        .filter(|s| !excused_strips.contains(&s.id))
        .collect();

    let mut stats = Stats::default();
    let mut report = String::new();

    let report_filename = format!(
        "reports/cc-{}-by-{}.txt",
        char_max,
        (format!("/{}", &args.model)).rsplit_once("/").unwrap().1
    );

    std::io::Write::write(
        &mut std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&report_filename)
            .unwrap(),
        b"",
    )
    .unwrap();

    for strip in small_strips.iter().take(30) {
        let (cost, avg_prob) = predict_strip(&strip, &model, &mut stats);
        println!(
            "{}: Estimated steps: {}. Avg. prob: {:.1}%",
            strip.id,
            cost,
            avg_prob * 100.0
        );

        if cost > 3000.0 {
            continue;
        }

        let search_res = search::practice_search(strip, model, Some(30000), &mut report);
        let result_msg = format!(
            "{} ==> ({:}/{:.1}%: [{}] {:} ({:.1}) {:.0}s)\n",
            strip.id,
            cost,
            avg_prob * 100.0,
            if search_res.found { "+" } else { " " },
            search_res.steps,
            search_res.steps as f64 / cost,
            search_res.seconds
        );
        print!("{}", result_msg);
        std::io::Write::write(
            &mut std::fs::OpenOptions::new()
                .write(true)
                .append(true)
                .open(&report_filename)
                .unwrap(),
            result_msg.as_bytes(),
        )
        .unwrap();
    }

    std::fs::write(
        format!(
            "reports/tok_scores/{}.cost-cal.txt",
            (format!("/{}", &args.model)).rsplit_once("/").unwrap().1
        ),
        stats.details,
    )
    .unwrap();
}

fn measure_costs(strips: &Vec<Strip>, model: &LlamaModel, args: &Args) {
    let representative_strips: Vec<&Strip> = strips
        .iter()
        .filter(|s| s.punchline.len() > 95 && s.punchline.len() < 120)
        .collect();

    // for strip in representative_strips.iter().progress() {
    //     println!(
    //         "{}: {} | {}\n-----------",
    //         strip.id,
    //         strip
    //             .leadup
    //             .chars()
    //             .skip(strip.leadup.len() - 150)
    //             .collect::<String>(),
    //         strip.punchline
    //     )
    // }

    let exemplars: Vec<usize> = vec![
        540, 2281, 2369, 2370, 1923, 2038, 811, 371, 1543, 2064, 1587, 2368, 951, 2297,
    ];
    // Some other proposed exemplars: 1448, 1699, 1551, 1392, 1959

    let mut stats = Stats::default();
    let mut costs = vec![];
    let mut avg_probs = vec![];
    for strip in representative_strips.iter().progress() {
        if !exemplars.contains(&strip.id) {
            continue;
        }
        let (cost, avg_prob) = predict_strip(&strip, &model, &mut stats);
        costs.push(cost);
        avg_probs.push(avg_prob);
    }
    std::fs::write(
        format!(
            "reports/tok_scores/{}.txt",
            (format!("/{}", &args.model)).rsplit_once("/").unwrap().1
        ),
        stats.details,
    )
    .unwrap();

    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "Costs: {}",
        percentile_table_2_digits!(&costs, (90, 75, 50, 25, 10), "{:.1e}", false)
    );

    println!(
        "Probs: {}",
        percentile_table_2_digits_percentage!(&avg_probs, (90, 75, 50, 25, 10), "{:.2}", true)
    );

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

    let mut prob_aheads = stats.prob_aheads;

    prob_aheads.sort_by(|a, b| a.partial_cmp(b).unwrap()); // higher is harder

    println!(
        "Calibration: {}",
        percentile_table_2_digits_percentage!(
            &prob_aheads,
            (50, 75, 80, 90, 95, 98, 99, 99.5),
            "{:.2}",
            false
        )
    );

    for tok_id in 0..10 {
        let mut tok_avg_probs = vec![];
        for sap in &stats.strip_avg_probs {
            tok_avg_probs.push(sap[tok_id]);
        }
        println!(
            "Token {} running average probability: {}",
            tok_id,
            percentile_table_2_digits_percentage!(
                &tok_avg_probs,
                (50, 25, 20, 10, 5, 2, 1),
                "{:.2}",
                false
            )
        );
    }

    println!();
    println!(
        "Average token time: {:.0}  Average truncation time: {:.0}  Average copy time: {:.0}",
        stats
            .tok_times
            .iter()
            .map(std::time::Duration::as_micros)
            .sum::<u128>() as f32
            / stats.tok_times.len() as f32,
        stats
            .truncate_times
            .iter()
            .map(std::time::Duration::as_micros)
            .sum::<u128>() as f32
            / stats.tok_times.len() as f32,
        stats
            .ctx_cpy_times
            .iter()
            .map(std::time::Duration::as_micros)
            .sum::<u128>() as f32
            / stats.tok_times.len() as f32
    );
}

fn main() {
    let args = Args::parse();
    init_tracing(&args);

    let mut model_params = LlamaParams::default();
    model_params.n_gpu_layers = if args.no_gpu { 0 } else { 1000000 };
    let model =
        LlamaModel::load_from_file(&args.model, model_params).expect("Could not load model");

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

    let strips = get_strips("corpus/strips.csv", &args.prompt_prefix);

    if args.calibrate_costs {
        calibrate_costs(&strips, &model, &args);
    } else {
        measure_costs(&strips, &model, &args);
    }

    // let mut toks_of_strips = vec![];
    // for strip in &strips {
    //     toks_of_strips.push(
    //         model
    //             .tokenize_bytes(&strip.leadup, true, false)
    //             .unwrap()
    //             .len()
    //             + model
    //                 .tokenize_bytes(&strip.punchline, true, false)
    //                 .unwrap()
    //                 .len(),
    //     );
    // }
    // toks_of_strips.sort();

    // println!(
    //     "(Percentile) tokens in a strip: (75) {}  (90) {}  (95) {}  (99) {}  (max) {}",
    //     toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.75).floor() as usize],
    //     toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.90).floor() as usize],
    //     toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.95).floor() as usize],
    //     toks_of_strips[((toks_of_strips.len() - 1) as f32 * 0.99).floor() as usize],
    //     toks_of_strips.last().unwrap(),
    // );

    // Result:
    // (Percentile) tokens in a strip: (75) 371  (90) 398  (95) 418  (99) 454  (max) 580
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
