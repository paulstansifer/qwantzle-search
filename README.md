# qwantzle-search

Uses an LLM to search for a solution to the Qwantzle (the punchline to [Dinosaur Comics strip 1663](https://www.qwantz.com/index.php?comic=1663)). The [solution tester](https://www.afifthofnothing.com/anacryptogram.html) has a rundown of the hints. I believe that the fact that the longest word was "fundamental" is an official hint, however, I can't find any *direct* evidence of this! Ryan North periodically deletes old tweets, so I'm hoping my memory is correct.

## How it works

For a given prefix, an LLM will give you the probabilities of each possible next token. This does a tree search over possible completions of a Dinosaur Comics strip, using those probabilities to determine which completions are most promising. Anything that uses unavailable letters is rule out, of course, but I also use a small regular old neural net to score the "quality" of the remaining letter pool (this keeps us from exploring branches where we're almost out of vowels, for example).

Thanks to the fact that pushing tokens onto the context and (with a little bit of low-level muckery with the KV cache) popping them off is basically free in llama.cpp; the time taken to reset the state of the model is almost nothing compared to the ~100ms it takes to generate logits from a particular prefix. This means that we can do an ordinary tree search, simply sticking all possible completions into a priority queue. The priority queue gets very large; it's necessary to prune it occasionally. If you want to run a search for a week, it needs to have at least a couple of gigabytes of RAM just to store the priority queue.

I've successfully unscrambled some Dinosaur Comics with punchlines that have 50-60 letters in a few hours this way. (I don't have too many examples, because I wanted to use as many comics as possible in the training set for fine-tuning!) The Qwantzle has 97 letters, so it may still be out of reach.

## How to use it

Install Cargo, and download an LLM in gguf format. ([I've got some suggestions](https://huggingface.co/paul-stansifer))

Then set some environment variables. On my Ubuntu machine, it's 
```
CC=/usr/bin/gcc-12 CXX=/usr/bin/gcc-12 CPATH=/usr/lib/gcc/x86_64-linux-gnu/12/include LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
```

Then do
```
RUSTFLAGS="-L $LIBRARY_PATH" cargo run --release -- -m <path to LLM> --search-one 1663
```

Why do you need to set `RUSTFLAGS` in that particular way? Faries insist on it, or they won't help. (Note to self: double-check this bit.)

You can hit Ctrl-C to stop a search and it will save progress. To resume, go

```
RUSTFLAGS="-L $LIBRARY_PATH" cargo run --release -- -m <path to LLM> --search-one /home/paul/.qwantzle/1663-in_progress.checkpoint
```

Hope your username is `paul`. (Note to self: check to see if anyone has a different first name than me.)
