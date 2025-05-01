# qwantzle-search

Uses an LLM to search for a solution to the Qwantzle (the punchline to [Dinosaur Comics strip 1663](https://www.qwantz.com/index.php?comic=1663)). The [solution tester](https://www.afifthofnothing.com/anacryptogram.html) has a rundown of the hints. I believe that the fact that the longest word was "fundamental" is an official hint, however, I can't find any *direct* evidence of this! Ryan North periodically deletes old tweets, so I'm hoping my memory is correct.

## How it works

Magic. (Note to self: write this section)

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