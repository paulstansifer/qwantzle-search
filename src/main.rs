use clap::Parser;
use indicatif::ProgressIterator;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::params::LlamaSamplerChainParams;
use llama_cpp_2::token_type::LlamaTokenAttr;
use llm::tok_to_str;
use search::{manual_search, SearchState};
use std::fmt::Write;
use std::io::{BufRead as _, Write as _};
use std::sync::atomic::AtomicBool;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

mod corpus;
mod llm;
mod pool;
mod remaining_letters_neural_net;
mod search;

use corpus::Strip;

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

    /// Just complete a prefix, using a sensible sampler
    #[arg(long)]
    complete: Option<String>,

    /// Score tokens for the given strip ID.
    #[arg(long)]
    tok_score: Option<usize>,

    // For tok-score, use a different suffix.
    #[arg(long)]
    suffix: Option<String>,

    /// Perform a search on the given strip ID, or loading a saved search
    #[arg(long)]
    search_one: Option<String>,

    #[arg(long)]
    search_manual: Option<usize>,

    /// Display the tokenization of the given string
    #[arg(long)]
    tokenize: Option<String>,

    /// Don't care about whether the theory of ties is respected
    #[arg(long, action = clap::ArgAction::SetTrue)]
    ignore_ties: bool,
}

#[derive(Default)]
struct Stats {
    details: String,
    tok_times: Vec<u128>,
    aheads: Vec<u32>,
    prob_aheads: Vec<f32>,
    probs: Vec<f32>,
    strip_avg_probs: Vec<Vec<f64>>,
}

fn percentile<T: PartialOrd + Copy + Default, U: num_traits::ToPrimitive>(
    samples: &Vec<T>,
    position: U,
    reverse: bool,
) -> T {
    if samples.is_empty() {
        return T::default();
    }
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
fn predict_strip(
    strip: &Strip,
    alt_punch: Option<&str>,
    model: &LlamaModel,
    stats: &mut Stats,
) -> (f64, f64) {
    let rlnn =
        remaining_letters_neural_net::LetterNet::new_from_file("corpus/letter_pool.safetensors")
            .unwrap();

    let punchline_to_examine = if let Some(alt_punch) = alt_punch {
        alt_punch
    } else {
        &strip.punchline
    };

    let toks_needed = model
        .str_to_token(&strip.leadup, llama_cpp_2::model::AddBos::Always)
        .unwrap()
        .len() as u32
        + model
            .str_to_token(&punchline_to_examine, llama_cpp_2::model::AddBos::Never)
            .unwrap()
            .len() as u32
        + 2;
    let mut sess = llm::Session::new(model, toks_needed);
    let mut candidates = sess.advance_and_predict_str(&strip.leadup, Some(0.99995));

    let (punch_toks, space) =
        llm::str_to_tokens_maybe_with_prefix_space(&punchline_to_examine, model);

    if space {
        writeln!(stats.details, "Padding the suffix with a space.").unwrap();
    } else {
        writeln!(stats.details, "Not padding the suffix with a space.").unwrap();
    };

    // Use the original punchline for the letter pool!
    let mut letter_pool = pool::LetterPool::from_text(&strip.punchline, /*ties*/ false);

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
    let mut filter_s = "   ".to_string();
    let mut chars_s = "   ".to_string();
    let mut tok_score_s = "   ".to_string();
    let mut ahead_s = "   ".to_string();
    let mut ahead_pv_s = "   ".to_string();
    let mut prob_ahead_s = "   ".to_string();
    let mut prob_s = "   ".to_string();

    let mut chars_i = 0.0;
    let mut overall_bonus: f64 = 1.0;

    let mut optimistic_cost = 1.0;
    let mut overall_probability: f64 = 1.0;

    let mut punchline: String = String::new();
    for tok in &punch_toks {
        stats.tok_times.push(sess.timers.predict_time);
        sess.timers.predict_time = 0;
        let mut ahead = 0;
        let mut ahead_and_pool_valid = 0;
        let mut prob_ahead = 0.0;
        let mut found_cand = None;
        for (cand_tok, prob) in candidates.iter() {
            if cand_tok == tok {
                found_cand = Some((cand_tok, prob));
                break;
            }
            ahead += 1;
            if letter_pool.has(*cand_tok, model) {
                ahead_and_pool_valid += 1;
            }

            prob_ahead += prob;
        }

        let tok_as_string = llm::tok_to_str(*tok, model);

        write!(tok_s, "{:>12}", tok_as_string).unwrap();

        let mut filter_bonus = 1.0;
        let mut char_bonus = 1.0;

        for _ in 0..tok_as_string.trim().bytes().len() {
            filter_bonus *=
                (1.0 / (0.05 + 0.55 * f32::max((80.0 - chars_i) / 80.0, 0.75))).powf(0.25);
            char_bonus *= (f32::max(0.0, ((100.0 - chars_i) / 100.0) * 2.0) + 3.0 + 1.0).powf(0.25);

            chars_i += 1.0;
        }

        write!(filter_s, "{:>12.2}", filter_bonus).unwrap();
        write!(chars_s, "{:>12.2}", char_bonus).unwrap();
        overall_bonus *= (filter_bonus * char_bonus) as f64;

        if let Some((_, prob)) = found_cand {
            write!(ahead_s, "{:>12}", ahead).unwrap();
            write!(ahead_pv_s, "{:>12}", ahead_and_pool_valid).unwrap();
            write!(prob_ahead_s, "{:>11.2}%", prob_ahead * 100.0).unwrap();
            write!(prob_s, "{:>11.3}%", prob * 100.0).unwrap();
            write!(
                tok_score_s,
                "{:>11.3}%",
                prob * filter_bonus as f64 * char_bonus as f64 * 100.0
            )
            .unwrap();
            stats.aheads.push(ahead);
            stats.probs.push(*prob as f32);
            stats.prob_aheads.push(prob_ahead as f32);
            overall_probability *= *prob as f64;
        } else {
            write!(ahead_s, " ---------- ").unwrap();
            write!(ahead_pv_s, " ---------- ").unwrap();
            write!(prob_ahead_s, " ---------- ").unwrap();
            write!(prob_s, " ---------- ").unwrap();
            write!(tok_score_s, " ---------- ").unwrap();
            stats.aheads.push(10000);
            stats.probs.push(0.0001);
            stats.prob_aheads.push(0.9999);
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

        candidates = sess.advance_and_predict(&[*tok], Some(0.99995));

        punchline.push(' ');
        punchline.push_str(&llm::tok_to_str(*tok, model));
        letter_pool.remove(*tok, &model);
    }
    let average_probability = f64::powf(overall_probability, 1.0 / punch_toks.len() as f64);

    write!(
        stats.details,
        "{tok_s}\n{prob_s}\n{filter_s}\n{chars_s}\n{tok_score_s}\n{ahead_s}\n{ahead_pv_s}\n{prob_ahead_s}\noptimistic cost: {:.2e}  average probability: {:.1}%  average tok time: {:.0} ",
        optimistic_cost,  average_probability * 100.0,
        stats.tok_times.iter().sum::<u128>() as f64 /
             (stats.tok_times.len()) as f64
    )
    .unwrap();

    let rlnn_score = rlnn.evaluate(&letter_pool) as f64;
    let overall_score = overall_probability * rlnn_score * overall_bonus;
    write!(
        stats.details,
        "score: {:.2e}% * {:.2}% * {overall_bonus:.2e} = {:.4}%\n",
        overall_probability * 100.0,
        rlnn_score * 100.0,
        overall_score * 100.0
    )
    .unwrap();

    if alt_punch.is_none() {
        assert!(letter_pool.size() == 0);
    } else {
        for (cand_tok, prob) in candidates.iter().take(10) {
            write!(
                stats.details,
                "'{}' {:.3}%   ",
                tok_to_str(*cand_tok, model),
                prob * 100.0,
            )
            .unwrap();
        }
        writeln!(stats.details).unwrap();
    }

    return (optimistic_cost, average_probability);
}

fn calibrate_costs(strips: &Vec<Strip>, words: &Vec<String>, model: &LlamaModel, args: &Args) {
    let char_min = 45;
    let char_max = 80;

    let get_strip = |id| strips.iter().find(|s| s.id == id).unwrap();

    // From way back in the first round of testing: they were in the training set then, but are
    // hold-outs in 3550!
    let round_1_exemplars = vec![
        1938, 876, 1218, 1575, 1737, 698, 1132, 1333, 1319, 392, 982, 1830, 1234, 1825, 1766, 1085,
        1567, 1374, 177, 2169, 1056, 1006, 2406, 2157, 1216, 1562, 1510, 787, 2363, 287,
    ]
    .into_iter()
    .map(get_strip)
    .collect::<Vec<_>>();

    // Hold-outs used in the first round of testing, retained for the 3550 models.
    let round_2_exemplars = vec![
        694, 2199, 819, 1977, 123, 1855, 1067, 1607, 1687, 353, 755, 2417, 2231, 1518, 1073, 1916,
        1907, 206, 606, 1199, 2463, 1323, 1055, 1650, 1898, 2188, 1375, 1395, 2390, 2337,
    ]
    .into_iter()
    .map(get_strip)
    .collect::<Vec<_>>();

    // Not necessarily holdouts in 3550.
    // let _quick_check_strips = [
    //     694, 1096, 205, 2231, 1518, 1916, 206, 1084, 1055, 1375, 1073,
    // ];

    let excused_strips = [
        2199, // Ought to get it, though all-caps makes it harder (has trouble with "BUSTING")
        1833, // Punchline has digits
        1743, // Formally weird; the leadup basically is just "Spring BREAAAAAAAAAAAA-"
        2258, // "double-down refill" is a very weird phrase
        1059, // Digits, being one-character tokens, make this one tough (1663 has no digits)
        886,  // "JODIE FOSTER" is in all caps and she's not mentioned or aluded to in the leadup
        2434, // Weird structure; I don't expect an LLM to figure this one out.
        2255, // "dillweed" as the second word is basically unguessable.
        1258, // TEMPORARY; WE SHOULD GET THIS
        1827, // TEMPORARY; WE SHOULD GET THIS
        1004, // TEMPORARY; WE SHOULD GET THIS
    ];

    let exemplars: Vec<&Strip> = round_1_exemplars
        .iter()
        .chain(round_2_exemplars.iter())
        .filter(|s| s.punchline.len() > char_min && s.punchline.len() <= char_max)
        .filter(|s| !s.punchline.contains("\n"))
        .filter(|s| !s.punchline.contains("(punchline)"))
        .filter(|s| !excused_strips.contains(&s.id))
        .cloned()
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

    let mut append_to_report = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(&report_filename)
        .unwrap();

    for strip in exemplars.iter().take(50) {
        let (cost, avg_prob) = predict_strip(&strip, None, &model, &mut stats);
        println!(
            "{}: Estimated steps: {}. Avg. prob: {:.1}%",
            strip.id,
            cost,
            avg_prob * 100.0
        );

        // // Add a couple of strips with known high estimates that are actually solveable quickly, to
        // // get more data:
        // if cost > 3_000_000.0 && strip.id != 694 && strip.id != 1055 {
        //     std::io::Write::write(
        //         &mut append_to_report,
        //         format!(
        //             "{: >4} {} ==> ({: >6}/{:.1}%)\n",
        //             strip.id,
        //             strip.punchline.len(),
        //             cost,
        //             avg_prob
        //         )
        //         .as_bytes(),
        //     )
        //     .unwrap();
        //     continue;
        // }

        let search_res = search::practice_search(strip, model, words, Some(250000), &mut report);
        if TIME_TO_QUIT.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let result_msg = format!(
            "{: >4} {} ==> ({: >6}/{:.1}%): [{}] {:} ({:.1}) {:.0}s\n",
            strip.id,
            strip.punchline.len(),
            cost,
            avg_prob * 100.0,
            if search_res.found { "+" } else { " " },
            search_res.steps,
            search_res.steps as f64 / cost,
            search_res.seconds
        );
        print!("{}", result_msg);
        std::io::Write::write(&mut append_to_report, result_msg.as_bytes()).unwrap();
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
        .filter(|s| s.punchline.len() > 70 && s.punchline.len() < 120)
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

    // let exemplars: Vec<usize> = vec![
    //     540, 2281, 2369, 2370, 1923, 2038, 811, 371, 1543, 2064, 1587, 2368, 951, 2297,
    // ];
    // Some other proposed exemplars: 1448, 1699, 1551, 1392, 1959

    let mut stats = Stats::default();
    let mut costs = vec![];
    let mut avg_probs = vec![];
    for strip in representative_strips.iter().progress() {
        let (cost, avg_prob) = predict_strip(&strip, None, &model, &mut stats);
        costs.push(cost);
        avg_probs.push(avg_prob);

        if TIME_TO_QUIT.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
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
            "probability score",
        ])
        .unwrap();

    for i in 0..stats.tok_times.len() {
        stats_csv
            .serialize((
                stats.tok_times[i],
                stats.aheads[i],
                stats.prob_aheads[i],
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

    print!("Token running average probs (5/2%): ");

    for tok_id in 0..10 {
        let mut tok_avg_probs = vec![];
        for sap in &stats.strip_avg_probs {
            if sap.len() <= tok_id {
                continue;
            }
            tok_avg_probs.push(sap[tok_id]);
        }

        print!(
            "{tok_id}: {:.2}%/{:.2}%   ",
            percentile(&tok_avg_probs, 5, false) * 100.0,
            percentile(&tok_avg_probs, 2, false) * 100.0
        );
    }

    println!();

    println!(
        "Average token time: {:.0}ms",
        stats.tok_times.iter().sum::<u128>() as f32 / stats.tok_times.len() as f32,
    );
}

fn complete(prefix: &str, max_new_toks: u32, model: &LlamaModel) {
    print!("{prefix}>>>");
    let prefix_tokens = model
        .str_to_token(prefix, llama_cpp_2::model::AddBos::Always)
        .unwrap();
    let n_toks = prefix_tokens.len() as u32 + max_new_toks;

    let params = llama_cpp_2::context::params::LlamaContextParams::default()
        .with_n_ctx(std::num::NonZero::new(n_toks));

    let mut ctx = {
        let _stderr_gag = gag::Gag::stderr().unwrap();
        model.new_context(&llm::BACKEND, params).unwrap()
    };

    let mut batch = LlamaBatch::new(n_toks as usize, 1);
    for (i, tok) in prefix_tokens.iter().enumerate() {
        batch
            .add(*tok, i as i32, &[0], i + 1 == prefix_tokens.len())
            .unwrap()
    }

    // This is suuuuuper ad-hoc.
    let mut sampler = llama_cpp_2::sampling::LlamaSampler::new(
        LlamaSamplerChainParams::default().with_no_perf(true),
    )
    .unwrap()
    .add_penalties(100, 9999, 9998, 150, 1.05, 1.05, 1.05, true, false)
    .add_mirostat_v2(0, 0.1, 5.0);

    for i in 0..max_new_toks {
        ctx.decode(&mut batch).expect("failed to eval");
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);

        sampler.accept(token);

        if token == model.token_eos() {
            eprintln!();
            break;
        }

        let output_str = model
            .token_to_str(token, llama_cpp_2::model::Special::Tokenize)
            .unwrap();

        print!("{output_str}");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        batch.clear();
        batch
            .add(token, prefix_tokens.len() as i32 + i as i32, &[0], true)
            .unwrap();
    }
}

static TIME_TO_QUIT: std::sync::atomic::AtomicBool = AtomicBool::new(false);

fn get_strip(id: usize, args: &Args) -> Strip {
    if id == 1663 {
        return Strip {
            leadup: String::from_utf8(std::fs::read("corpus/1663-prefix.txt").unwrap()).unwrap(),
            punchline: "ttttttttttttooooooooooeeeeeeeeaaaaaaallllllnnnnnnuuuuuuiiiiisssssdddddhhhhhyyyyyIIrrrfffbbwwkcmvg:,!!".to_owned(),
            id: 1663,
        };
    }
    let strips = corpus::get_strips("corpus/validation_strips.csv", &args.prompt_prefix);
    for strip in strips {
        if strip.id == id {
            return strip;
        }
    }
    panic!("Strip with id {id} not found!");
}

fn main() {
    use signal_hook::{
        consts::{SIGINT, SIGTERM},
        iterator::Signals,
    };
    let args = Args::parse();
    init_tracing(&args); // TODO: no longer has any effect, I think.

    let mut signals = Signals::new(&[SIGTERM, SIGINT]).unwrap();

    std::thread::spawn(move || {
        for _ in signals.forever() {
            TIME_TO_QUIT.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });

    let model = llm::model_from_gguf(&args.model, !args.no_gpu);

    // Can use "corpus/dictionary_filter.txt", but it's not worth it.
    let words = corpus::get_words("corpus/allowed_words.txt", None);

    if args.calibrate_costs {
        let strips = corpus::get_strips("corpus/validation_strips.csv", &args.prompt_prefix);

        calibrate_costs(&strips, &words, &model, &args);
    } else if let Some(ref pfx) = args.complete {
        if let Ok(id) = pfx.parse::<usize>() {
            complete(&get_strip(id, &args).leadup, 200, &model);
        } else {
            complete(&pfx, 200, &model);
        }
    } else if let Some(id) = args.tok_score {
        let mut stats = Stats::default();
        let strip = get_strip(id, &args);
        predict_strip(&strip, args.suffix.as_deref(), &model, &mut stats);
        println!("{}", stats.details);
    } else if let Some(ref search) = args.search_one {
        if let Ok(id) = search.parse::<usize>() {
            let hints = if id == 1663 {
                search::Hints::for_1663(&words, !args.ignore_ties, &model)
            } else {
                search::Hints::from_strip(&get_strip(id, &args), &words, !args.ignore_ties, &model)
            };
            let mut search = SearchState::new(&model, hints, None);
            search.search();
        } else {
            let mut search = SearchState::load(&search, &model);
            search.search();
        }
    } else if let Some(id) = args.search_manual {
        let hints = if id == 1663 {
            search::Hints::for_1663(&words, !args.ignore_ties, &model)
        } else {
            search::Hints::from_strip(&get_strip(id, &args), &words, !args.ignore_ties, &model)
        };
        println!("{}", &hints.leadup);

        print!("> ");
        std::io::stdout().flush().unwrap();
        let reader = std::io::BufReader::new(std::io::stdin());
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    manual_search(&model, hints.clone(), &line);
                }
                Err(err) => {
                    panic!("{:?}", err);
                }
            }
            print!("> ");
            std::io::stdout().flush().unwrap();
        }
    } else if let Some(s) = args.tokenize {
        let toks = llm::str_to_tokens(&s, &model);
        let mut s_top = String::new();
        let mut s_bot = String::new();
        for tok in toks {
            let tok_str = tok_to_str(tok, &model).replace("\n", "\\n");
            s_top += &format!("'{}' ", tok_str);
            s_bot += &format!(" {}", tok.0);

            if model.token_attr(tok).contains(LlamaTokenAttr::Control) {
                s_bot += "[C]"
            }

            while s_bot.len() < s_top.len() {
                s_bot += " ";
            }
            while s_top.len() < s_bot.len() {
                s_top += " ";
            }
        }
        print!("{s_top}\n{s_bot}\n");
    } else {
        // Strips withheld from the 3550 corpus. Pre-3550 models may perform better because of memorization.
        let strips = corpus::get_strips("corpus/validation_strips.csv", &args.prompt_prefix);
        measure_costs(&strips, &model, &args);
    }
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
