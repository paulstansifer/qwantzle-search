use std::rc::Rc;

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Strip {
    pub id: usize,
    pub prompt: Rc<String>,
    pub leadup: String,
    pub punchline: String,
}

impl Strip {
    pub fn context(&self) -> String {
        format!(
            "{}{}{}",
            self.prompt,
            if self.prompt.ends_with("\n") {
                ""
            } else {
                "\n"
            },
            self.leadup
        )
    }
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

pub fn get_strips(path: &str, prompt: Rc<String>) -> Vec<Strip> {
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
                    id: str::parse::<usize>(record.get(0).unwrap()).unwrap(),
                    prompt: prompt.clone(),
                    leadup: format!("{}\n{}: {}", leadup, speaker, punchline_first_word),
                    punchline: punchline_rest.to_owned(),
                };

                res.push(strip);
            }
        }
    }
    return res;
}
