# Frontier admission v6 retrieval receipts

The v6 task was pushed as version 7 with protocol identifier
`ai-race-frontier-admission-v6`. The server completed the Gemini 3 Flash run,
but the local downloader failed before writing any output artifact:

```text
[Errno 2] No such file or directory: ...
ai-race-frontier-admission-run_id_Run_1_google_gemini-3-flash-preview.run.json
```

Because the raw responses, completed manifest, and run receipt were not
retrievable, this route remains `blocked` and is not promoted to admission
evidence. The same retrieval gate will be applied to every other route. v5
artifacts remain in their original directory and are not overwritten.
