use regex;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Strip {
    pub id: usize,
    pub leadup: String,
    pub punchline: String,
}

pub fn get_words(allowed: &str, forbidden: Option<&str>) -> Vec<String> {
    let mut non_words = std::collections::HashSet::new();

    if let Some(forbidden) = forbidden {
        let forbidden_file = std::fs::File::open(forbidden).unwrap();
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b' ')
            .from_reader(forbidden_file);
        for result in reader.records() {
            non_words.insert(result.unwrap().get(0).unwrap().trim().to_string());
        }
    }

    let file = std::fs::File::open(allowed).unwrap();
    let mut reader = csv::ReaderBuilder::new().delimiter(b' ').from_reader(file);

    let mut res = vec![];
    for result in reader.records() {
        let record = result.unwrap();
        let word = record.get(1).unwrap().trim().to_string();

        if !non_words.contains(&word) {
            res.push(word);
        }
    }
    return res;
}

pub fn get_strips(path: &str, prompt_prefix: &str) -> Vec<Strip> {
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
                let prompt_prefix_formatted = if prompt_prefix.is_empty() {
                    "".to_owned()
                } else if prompt_prefix == "l" {
                    let mut words: Vec<&str> = regex::Regex::new(r"\W+")
                        .unwrap()
                        .split(punchline_rest)
                        .collect();
                    words.sort_by(|a, b| a.len().cmp(&b.len()));
                    format!(
                        "The longest word in the punchline is \"{}\".\n---\n",
                        words.last().unwrap()
                    )
                } else {
                    format!("{}\n---\n", prompt_prefix)
                };

                let strip = Strip {
                    id: str::parse::<usize>(record.get(0).unwrap()).unwrap(),
                    leadup: format!(
                        "{}{}\n{}: {}",
                        prompt_prefix_formatted, leadup, speaker, punchline_first_word
                    ),
                    punchline: punchline_rest.to_owned(),
                };

                res.push(strip);
            }
        }
    }
    return res;
}
