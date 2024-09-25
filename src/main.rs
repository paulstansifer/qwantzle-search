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

    /// String to prepend to every strip. Use "l" to add a message about what the longest word in
    /// the punchline is.
    #[arg(short, long, default_value(""))]
    prompt_prefix: String,
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

    let punch_toks = ctx
        .model()
        .tokenize_bytes(&strip.punchline, false, false)
        .unwrap();

    {
        // Determine how long copy and truncation take.
        let before_cpy = std::time::Instant::now();
        let mut copied_ctx = ctx.deep_copy().unwrap();
        stats.ctx_cpy_times.push(before_cpy.elapsed());

        let before_trn = std::time::Instant::now();
        copied_ctx
            .truncate_context(ctx.context_size() - 10)
            .unwrap();
        stats.truncate_times.push(before_trn.elapsed());
    }

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

    let mut tok_s = "   ".to_string();
    let mut ahead_s = "   ".to_string();
    let mut prob_ahead_s = "   ".to_string();
    let mut logit_s = "   ".to_string();
    let mut prob_s = "   ".to_string();

    let mut optimistic_cost = 1.0;
    let mut overall_probability: f64 = 1.0;

    let mut punchline: String = String::new();
    for (tok_i, tok) in punch_toks.iter().enumerate() {
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
            overall_probability *= cand.p as f64;
        } else {
            write!(ahead_s, " -------- ").unwrap();
            write!(prob_ahead_s, " -------- ").unwrap();
            write!(logit_s, " -------- ").unwrap();
            write!(prob_s, " -------- ").unwrap();
            stats.aheads.push(10000);
            stats.probs.push(0.0001);
            stats.prob_aheads.push(0.9999);
            stats.logits.push(-9999.9);
            overall_probability *= 0.0001 as f64;
        }

        let anagram_restriction = 0.8 - (0.75 * (tok_i as f64 / (punch_toks.len() as f64)));

        optimistic_cost *= (*stats.aheads.last().unwrap() as f64 * anagram_restriction) + 1.0;

        ctx.advance_context_with_tokens(&[*tok])
            .expect("Advancement error!");
        punchline.push_str(&ctx.model().decode_tokens([*tok].iter()));
    }

    let average_probability = f64::powf(overall_probability, 1.0 / punch_toks.len() as f64);

    write!(
        stats.details,
        "{tok_s}\n{logit_s}\n{prob_s}\n{ahead_s}\n{prob_ahead_s}\noptimistic cost: {:.2e}  average probability: {:.3e}  average tok time: {:.0}\n",
        optimistic_cost,  average_probability,
        stats.tok_times.iter().map(|d| d.as_micros()).sum::<u128>() as f64 /
             (stats.tok_times.len()) as f64
    )
    .unwrap();

    return (optimistic_cost, average_probability);
}

fn main() {
    let args = Args::parse();
    init_tracing(&args);

    let mut model_params = LlamaParams::default();
    model_params.n_gpu_layers = if args.no_gpu { 0 } else { 1000000 };
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

    let strips = get_strips("strips.csv", &args.prompt_prefix);

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

    let representative_strips: Vec<&Strip> = strips
        .iter()
        .filter(|s| s.punchline.len() > 95 && s.punchline.len() < 120)
        .collect();

    // for strip in representative_strips.iter().progress() {
    //     println!(
    //         "{}: {} | {}",
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
    std::fs::write("details.txt", stats.details).unwrap();

    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "(Percentile) costs: (90) {:.1e}  (75) {:.1e}  (50) {:.1e}  (25) {:.1e}  (10) {:.1e}",
        costs[((costs.len() - 1) as f32 * 0.90).floor() as usize],
        costs[((costs.len() - 1) as f32 * 0.75).floor() as usize],
        costs[((costs.len() - 1) as f32 * 0.50).floor() as usize],
        costs[((costs.len() - 1) as f32 * 0.25).floor() as usize],
        costs[((costs.len() - 1) as f32 * 0.10).floor() as usize],
    );

    avg_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!(
        "(Percentile) avg_probs: (90) {:.2}%  (75) {:.2}%  (50) {:.2}%  (25) {:.2}%  (10) {:.2}%",
        avg_probs[((avg_probs.len() - 1) as f32 * 0.90).floor() as usize] * 100.0,
        avg_probs[((avg_probs.len() - 1) as f32 * 0.75).floor() as usize] * 100.0,
        avg_probs[((avg_probs.len() - 1) as f32 * 0.50).floor() as usize] * 100.0,
        avg_probs[((avg_probs.len() - 1) as f32 * 0.25).floor() as usize] * 100.0,
        avg_probs[((avg_probs.len() - 1) as f32 * 0.10).floor() as usize] * 100.0,
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

    print!("Calibration: ");
    for p_limit in [0.5, 0.75, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995] {
        let prob_aheads_limit =
            prob_aheads[((prob_aheads.len() - 1) as f32 * p_limit).floor() as usize];

        print!(
            "({:.2}%) {:.2}%  ",
            p_limit * 100.0,
            prob_aheads_limit * 100.0
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
