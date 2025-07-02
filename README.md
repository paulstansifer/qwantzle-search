# qwantzle-search

The punchline to [Dinosaur Comics strip 1663](https://www.qwantz.com/index.php?comic=1663) was given in anagram form. It's now known as the "Qwantzle", and people have been trying to unscramble it since it was published in 2010. 

## How it works

For a given prefix, an LLM will give you the probabilities of each possible next token; these probabilites turn out to be pretty well-calibrated in practice. `qwantzle-search` does a tree search over possible completions of a Dinosaur Comics strip, using those probabilities to determine which token sequences are the most promising. Anything that uses unavailable letters is ruled out, of course, but I also use a small regular old neural net to score the "quality" of the remaining letter pool (this keeps us from exploring branches where we're almost out of vowels, for example).

Thanks to the fact that pushing tokens onto the context and (with a little bit of low-level muckery with the KV cache) popping them off is basically free in llama.cpp; the time taken to reset the state of the model is almost nothing compared to the ~100ms it takes to generate logits from a particular prefix. This means that we can do an ordinary tree search, simply sticking all possible completions into a priority queue. The priority queue gets very large; it's necessary to prune it occasionally. If you want to run a search for a week, it needs to have at least a couple of gigabytes of RAM just to store the priority queue.

I've successfully unscrambled some Dinosaur Comics with punchlines that have 50-60 letters in a few hours this way. (I don't have too many examples, because I wanted to use as many comics as possible in the training set for fine-tuning!) The Qwantzle has 97 letters, so it may still be out of reach; I've given it roughly week-long runs and not found an answer yet.

## Constraints

The [solution tester](https://www.afifthofnothing.com/anacryptogram.html) has a rundown of the hints that Ryan North has provided; the solver implements all of them (and generates the corresponding hints when test-solving other comics). I believe that the fact that the longest word was "fundamental" is an official hint, however, I can't find any direct evidence of this! Ryan North periodically deletes old tweets, so my best guess it that it got deleted.

### The theory of ties

Some people have noticed that the letters that occur the same number of times in the Qwantzle punchline are not given in alphabetical order. If they are listed in order of first occurrence, it's a huge hint (much larger than most of the ones Ryan North provided explicitly, based on my experiments with other comics with shorter punchlines). The solver keeps two priority queues, one for prefixes whose first letter occurences respect that ordering and one for ones that don't. It ensures that at least a certain proportion of its time is spent examining prefixes that respect the "theory of ties". (My general testing of other strips has not assumed the theory of ties, however, because it is such a huge help it makes a lot of comics trivial to unscramble.)

## How to use it

Install Cargo, and download an LLM in gguf format. ([I've got some LLMs fine-tuned on the Dinosaur Comics corpus](https://huggingface.co/paul-stansifer)).

Then set some environment variables. On my Ubuntu machine, it's 
```
CC=/usr/bin/gcc-12 CXX=/usr/bin/gcc-12 CPATH=/usr/lib/gcc/x86_64-linux-gnu/12/include LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
```

Then do
```
RUSTFLAGS="-L $LIBRARY_PATH" cargo run --release -- -m <path to LLM> --search-one 1663
```

Why do you need to set `RUSTFLAGS` in that particular way? Faries insist on it, or they won't help make the LLM go. (Note to self: double-check this bit.)

You can hit Ctrl-C to stop a search and it will save progress. To resume, go

```
RUSTFLAGS="-L $LIBRARY_PATH" cargo run --release -- -m <path to LLM> --search-one /home/paul/.qwantzle/1663-in_progress.checkpoint
```

Hope your username is `paul`. (Note to self: check to see if anyone has a different first name than me.)
