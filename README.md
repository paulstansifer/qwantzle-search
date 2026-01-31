# Idiom Scour/Scan

Dinosaur Comics is a long-running webcomic by Ryan North, in which the art is identical in every strip; only the words change. In 2010, [strip 1663](https://www.qwantz.com/index.php?comic=1663) discussed a brief fad among scientists in the 1600s of claiming priority by publishing anagrams of their ideas, an early form of [commitment scheme](https://en.wikipedia.org/wiki/Commitment_scheme). The punchline to the strip is presented in anagram form: we know the letters, but not what order they go in. It's now known as the "Qwantzle", and people have been trying to unscramble it since it was published. 

Idiom Scour/Scan is a tool to unscramble anagrams by doing a search that broadly examines possible sentence beginnings and deeply investigates promising leads, all while trying to fit the context and authorial voice.  (Also, "Idiom Scour/Scan" is an anagram of "Dinosaur Comics".)

## How it works

For a given prefix, an LLM will give you the probabilities of each possible next token; these probabilites turn out to be pretty well-calibrated in practice. `idiom-scour-scan` does a tree search over possible completions of a Dinosaur Comics strip, using those probabilities to determine which token sequences are the most promising. Anything that uses unavailable letters is ruled out, of course, but I also use a small regular old neural net to score the "quality" of the remaining letter pool (this keeps us from exploring branches where we're almost out of vowels, for example).

Pushing tokens onto the context and (with a little bit of low-level muckery with the KV cache) popping them off is basically free in llama.cpp: the time taken to reset the state of the model is almost nothing compared to the ~100ms it takes to generate logits from a particular prefix[^1]. This means that we can do an ordinary tree search, simply sticking all possible completions into a priority queue. The priority queue gets very large; it's necessary to prune it occasionally. If you want to run a search for a week, it needs to have at least a couple of gigabytes of RAM just to store the priority queue.

[^1]: I believe there's some caching under the hood, because presumably pushing tokens should have a cost. In this case, however, the last token is the only one that's never been seen before.

Most Dinosaur Comics with punchlines that have up to ~65 letters can be unscrambled in a few hours this way. (See "reports/README.md" for how to read the tests.) (I don't have too many examples, because I wanted to use as many comics as possible in the training set for fine-tuning!) The Qwantzle has 97 letters, so it may still be out of reach; I've given it roughly week-long runs and not found an answer yet.

## Constraints

The [solution tester](https://www.afifthofnothing.com/anacryptogram.html) has a rundown of the hints that Ryan North has provided; the solver implements all of them (and generates the corresponding hints when test-solving other comics). It omits that "fundamental" is the longest word; this was [revealed somewhat later](https://www.qwantz.com/index.php?comic=1695), and for a long time I was wondering whether I had been imagining it!

### The theory of ties

Some people have noticed that the letters that occur the same number of times in the Qwantzle punchline are not given in alphabetical order. If they are listed in order of first occurrence, it's a huge hint (much larger than most of the ones Ryan North provided explicitly, based on my experiments with other comics with shorter punchlines). The solver keeps two priority queues, one for prefixes whose first letter occurences respect that ordering and one for ones that don't. It ensures that at least a certain proportion of its time is spent examining prefixes that respect the "theory of ties". (My general testing of other strips has not assumed the theory of ties, however: it is such a huge help it makes a lot of comics trivial to unscramble.)

## How to use it

Install Cargo, and download an LLM in gguf format. ([I've got some LLMs fine-tuned on the Dinosaur Comics corpus](https://huggingface.co/paul-stansifer)).

Then do `cargo build`, see what goes wrong, and mess with `.cargo/config.toml` to get it to work. I believe you specifically need GCC 12 in order to compile `llama.cpp`.

Then do
```
cargo run --release -- -m <path to LLM> --search-one 1663
```

You can hit Ctrl-C to stop a search and it will save progress. To resume, go

```
cargo run --release -- -m <path to LLM> --search-one /home/paul/.qwantzle/1663-in_progress.checkpoint
```

Hope your username is `paul`. (Note to self: check to see if anyone has a different first name than me.)
