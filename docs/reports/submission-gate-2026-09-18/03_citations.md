# Gate 03: Citation verification

Date: 2026-09-18. Scope: all 53 entries of `paper/references.bib`, checked against `paper/main.tex` and `paper/supplementary.tex`. The audit was read only; no repository file was edited. The table for each entry is in `03_citations.csv`.

Sources used: Crossref REST API (DOIs), arXiv export API (every eprint id), ACL Anthology `.bib` records, PMLR and ICLR proceedings pages (citation meta tags), IOS Press ebook page, DataCite (Zenodo DOI), Semantic Scholar batch API (preprint to venue), and web search. The retained sources were used for claim checks: `references/papers/sources/arXiv-2607.26034v1/paper.tex` and the JAIR 12225 text. arXiv sources were downloaded to the scratchpad for Akata et al. (2305.16867v2), Payne and Alloui-Cros (2507.02618v1) and Pal et al. (2601.09849v1), and ACL PDFs for Cao et al., Giorgi et al. and SocioBench.

`paper/CITATION_CHANGELOG.md` was read first. The items it already fixed (the Sclar, Pezeshkpour, Wei, Salinas, Zheng, Aher, del Rio-Chanona, Akata and Herr claim wording) are not reported again. The only exception is Salinas, whose published venue the changelog missed.

## Summary counts

| Item | Count |
|---|---|
| Bib entries | 53 |
| Verified fully (`yes`) | 45 |
| Partly verified or with wrong or outdated metadata (`partial`) | 7 |
| Not verifiable (`no`) | 1 (uncited) |
| Fabricated entries found | 0 |
| Cited in main only / supp only / both / never | 19 / 3 / 14 / 17 |
| `\cite` keys with no bib entry | 0 |
| Duplicate works | 1 pair |
| Preprints that now have a published version | 2 (both cited) |
| Claim-support problems among the main-text citations | 3 substantive, 3 minor |

## 1. Fabricated or unverifiable entries

No cited entry is fabricated. Every cited work exists, and its title, authors and year match a primary record.

**`minhEtAlValiditySurvey` (uncited): unverifiable.** It is an `@unpublished` manuscript with `year = {n.d.}`. arXiv title search finds no record, and it has no DOI. Because it is never cited, BibTeX prints nothing for it. Delete it, or keep it out of the bib until a public preprint exists.

## 2. Wrong metadata

### 2.1 Cited entries (these change the printed reference list)

**`to_regulate_or_not`: the title is wrong in print.** The bib has `To regulate or not: A social dynamics analysis of an idealised ai race`, and `main.bbl` prints "...an idealised ai race". The official title (Crossref, JAIR) is *To Regulate or Not: A Social Dynamics Analysis of an Idealised AI Race*.

```bibtex
@article{to_regulate_or_not,
  title     = {To Regulate or Not: A Social Dynamics Analysis of an Idealised {AI} Race},
  author    = {Han, The Anh and Pereira, Lu{\'i}s Moniz and Santos, Francisco C. and Lenaerts, Tom},
  journal   = {Journal of Artificial Intelligence Research},
  volume    = {69},
  pages     = {881--921},
  year      = {2020},
  publisher = {AI Access Foundation},
  issn      = {1076-9757},
  doi       = {10.1613/jair.1.12225}
}
```

**`sclarPromptFormatting2024`: page numbers missing.** BibTeX warns about this in both `main.blg` and `supplementary.blg`. The ICLR proceedings page gives volume 2024, pp. 25055–25083.

```bibtex
@inproceedings{sclarPromptFormatting2024,
  author    = {Sclar, Melanie and Choi, Yejin and Tsvetkov, Yulia and Suhr, Alane},
  title     = {Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I Learned to Start Worrying About Prompt Formatting},
  booktitle = {International Conference on Learning Representations},
  volume    = {2024},
  pages     = {25055--25083},
  year      = {2024},
  address   = {Vienna, Austria},
  publisher = {OpenReview.net},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html}
}
```

**`suhSurveyDistributions2025`: booktitle abbreviated.** The metadata is otherwise correct, but the paper is Cao et al., not Suh et al., so the key is misleading. The key can stay because renaming it means editing `main.tex`.

```bibtex
@inproceedings{suhSurveyDistributions2025,
  author    = {Cao, Yong and Liu, Haijiang and Arora, Arnav and Augenstein, Isabelle and R{\"o}ttger, Paul and Hershcovich, Daniel},
  title     = {Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages     = {3141--3154},
  year      = {2025},
  address   = {Albuquerque, New Mexico},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.naacl-long.162},
  url       = {https://aclanthology.org/2025.naacl-long.162/}
}
```

`wangHumanSubjectivity2024` has the same problem: the authors are Giorgi et al. and the metadata is correct. The key is cosmetic and needs no fix.

**Formatting fixes (small; what is printed does not change or barely changes):**

- `ai_race_for_strategic_advantage`, `out_of_one` and `synthetic_replacement_human` use a Unicode en dash in `pages`. Replace it with `--`: `pages = {36--40}`, `pages = {337--351}` and `pages = {401--416}`.
- `risk-taking_touraments`, `risk_taking_in_competition` and `competition_and_risk-taking` store a full URL in `doi`. It renders correctly under ACM-Reference-Format now, but should hold the bare DOI: `doi = {10.1016/j.joep.2009.03.009}`, `doi = {10.1016/j.jcorpfin.2014.05.003}` and `doi = {10.1016/j.euroecorev.2023.104592}`.

Page-number note: for the three EMNLP 2025 papers (`wangSocioBench2025`, `shiEtAl2025Route`, `aradEtAl2025Steering`), Crossref's pages are 11 higher than the ACL Anthology's. The bib follows the Anthology, which is authoritative. No change is needed.

### 2.2 Uncited entries (fix only if they will be cited)

```bibtex
@inproceedings{han2019modelling,
  title     = {Modelling and Influencing the {AI} Bidding War: A Research Agenda},
  author    = {Han, The Anh and Pereira, Lu{\'\i}s Moniz and Lenaerts, Tom},
  booktitle = {Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society},
  pages     = {5--11},
  year      = {2019},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3306618.3314265}
}

@article{han2022voluntary,
  title     = {Voluntary safety commitments provide an escape from over-regulation in {AI} development},
  author    = {Han, The Anh and Lenaerts, Tom and Santos, Francisco C. and Pereira, Lu{\'\i}s Moniz},
  journal   = {Technology in Society},
  volume    = {68},
  pages     = {101843},
  year      = {2022},
  publisher = {Elsevier},
  doi       = {10.1016/j.techsoc.2021.101843}
}

@article{gruetzemacher2025strategic,
  title     = {Strategic insights from simulation gaming of {AI} race dynamics},
  author    = {Gruetzemacher, Ross and Avin, Shahar and Fox, James and Saeri, Alexander K.},
  journal   = {Futures},
  volume    = {167},
  pages     = {103563},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.futures.2025.103563}
}

@article{bengio2024managing,
  title     = {Managing extreme {AI} risks amid rapid progress},
  author    = {Bengio, Yoshua and Hinton, Geoffrey and Yao, Andrew and Song, Dawn and Abbeel, Pieter and Darrell, Trevor and Harari, Yuval Noah and Zhang, Ya-Qin and Xue, Lan and Shalev-Shwartz, Shai and others},
  journal   = {Science},
  volume    = {384},
  number    = {6698},
  pages     = {842--845},
  year      = {2024},
  publisher = {American Association for the Advancement of Science},
  doi       = {10.1126/science.adn0117}
}

@inproceedings{piattiGovSim2024,
  title     = {Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of {LLM} Agents},
  author    = {Piatti, Giorgio and Jin, Zhijing and Kleiman-Weiner, Max and Sch{\"o}lkopf, Bernhard and Sachan, Mrinmaya and Mihalcea, Rada},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {111715--111759},
  year      = {2024},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/079017-3548}
}
```

- **`ord2020precipice`**: the publisher "Hachette UK, London" looks like a Google Books artefact. A review indexed in Crossref (Pro-Fil 21(1), doi 10.5817/pf20-1-2120) gives "Bloomsbury Publishing, 2020". The publisher's own page was not checked, so treat this as partly confirmed. If the entry is cited, use `publisher = {Bloomsbury Publishing}, address = {London}`.
- **`mousaviDavoudiEtAl2026`**: the entry matches arXiv v1. v2 (2026-09-10) changes the author order to Mousavi Davoudi, Gharagozlou, Amiri-Margavi, Gholami Davodi, Hasani Balyani. If it is cited, either cite v2 with that order or keep `version = {1}`.
- **`liEtAl2026Fast`**: the entry matches v2 (2026-07-20). v1 (2025) had the title "Training Superior Sparse Autoencoders for Instruct Models". No venue was found.
- **`fernandezDomingos2020Egttools`**: see §5. It should be cited, and its metadata needs correcting.

## 3. Preprint to published upgrades (both cited)

**`salinasButterfly2024`**: published in Findings of ACL 2024. Confirmed from the ACL Anthology `.bib` and Crossref.

```bibtex
@inproceedings{salinasButterfly2024,
  author    = {Salinas, Abel and Morstatter, Fred},
  title     = {The Butterfly Effect of Altering Prompts: How Small Changes and Jailbreaks Affect Large Language Model Performance},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages     = {4629--4651},
  year      = {2024},
  publisher = {Association for Computational Linguistics},
  address   = {Bangkok, Thailand},
  doi       = {10.18653/v1/2024.findings-acl.275},
  url       = {https://aclanthology.org/2024.findings-acl.275/}
}
```

**`buscemiFAIRGAME2025`**: published at ECAI 2025. Confirmed from the IOS Press page (vol. 413, pp. 4097–4104) and Crossref (DOI 10.3233/FAIA251300, ISBN 9781643686318). The claim at main.tex:429 ("across models, languages and persona conditions against game-theoretic predictions") is supported by the ECAI abstract, so the v5 preprint is not needed. The IOS page did not list the editors, so they are left out.

```bibtex
@inproceedings{buscemiFAIRGAME2025,
  author    = {Buscemi, Alessio and Proverbio, Daniele and Di Stefano, Alessandro and Han, The Anh and Castignani, German and Li{\`o}, Pietro},
  title     = {{FAIRGAME}: A Framework for {AI} Agents Bias Recognition Using Game Theory},
  booktitle = {ECAI 2025},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {413},
  pages     = {4097--4104},
  year      = {2025},
  publisher = {IOS Press},
  doi       = {10.3233/FAIA251300}
}
```

No other cited preprint has a published version. Checked on Semantic Scholar and by web search: Askell et al., Fernández Domingos and Han, Herr et al. (an earlier ICML 2024 workshop paper exists under a different title; it is not the same version, so keep the arXiv entry), Robinson and Burden, del Rio-Chanona et al., Pal et al., Zheng et al. 2025, Yao et al. (submitted to CDC 2026), both Huynh et al. papers, Payne and Alloui-Cros, and Lekeas and Stamatopoulos.

## 4. Claim support (main text, the ten most important citations)

| # | Citation, location | Verdict | Supporting passage |
|---|---|---|---|
| 1 | `falling_behind_unsafe`, main.tex:383–389, 571–574 | Supported | Abstract: "Neither the pre-registered comparison between risk levels nor the role of elicited risk preferences was supported by the data ... participants are more likely to choose Unsafe after their opponent does so, being ahead reduces Unsafe play while falling behind increases it, and first-round choices predict later behaviour." paper.tex:1024: "AU at p_r^max=0.1, CAS at p_r^max=0.6, and CS at p_r^max=0.9". |
| 2 | `to_regulate_or_not`, main.tex:554 (group-count rule) | Supported | JAIR Appendix, N-player definition: "In a group of where k players choosing SAFE and (N−k) choosing UNSAFE, the payoffs ... are π(k)SAFE ... π(k)UNSAFE". |
| 3 | `playing_repeated_games_with_llms`, main.tex:401–404 | Supported | Source main.tex:212: "GPT-4 seemingly *could* predict the alternating patterns but instead just did not act in accordance with the resulting convention." |
| 4 | `using_llm_to_simulate`, main.tex:405 | Supported | PMLR abstract: "In the first three TEs, the existing findings were replicated ... the last TE reveals a 'hyper-accuracy distortion'". |
| 5 | `synthetic_replacement_human`, main.tex:411 | Supported | Abstract: "The average scores generated by ChatGPT correspond closely to the averages ... Nevertheless ... there is less variation in responses than in the real surveys". |
| 6 | `herrStrategicBias2024`, main.tex:418 | Supported | Abstract: "affected by at least one of the following systematic biases: positional bias, payoff bias, or behavioural bias." |
| 7 | `buscemiFAIRGAME2025`, `huynhUnderstanding2025`, `huynhPayoffScaling2026`, main.tex:429–432 | Supported | 2601.19082 abstract: "as stakes grow, evolutionary theory predicts that defection should take over the population, yet LLMs move in the opposite direction, becoming more cooperative". 2512.07462 abstract: "a payoff-scaled Prisoners Dilemma ... multi-agent Public Goods Game ... supervised classification models on canonical repeated-game strategies". |
| 8 | Heterogeneity list, main.tex:437–440 | **Partly unsupported** | See 4.1. |
| 9 | `anthisSocialSimulation2025`, main.tex:441–443; `pal`/`zheng`/`yao`/`lu`, main.tex:444–448 | Supported | Anthis: "LLM social simulations can already be used for pilot and exploratory studies". Zheng: "they apply these rules more rigidly". Yao: "Rather than converging to Nash equilibria, we find that LLM agents tend to cooperate". Lu: "prompt-based LLMs ... achieve only 11.86% accuracy in generating human actions". Pal: "we explore how LLMs adapt their strategies to changes in parameter values ... We also study the effect of different framings ... none of them exhibit full consistency". |
| 10 | `lekeasNashSuppression2026`, `palCooperation2026`, `payneStrategicIntelligence2025`, main.tex:472–475 | **Pal unsupported**; Lekeas and Payne supported | See 4.2. Lekeas abstract: "The cross-play experiments reveal three phenomena invisible in self-play ... who moves first determines which Nash equilibrium the system reaches." |

### 4.1 Heterogeneity sentence (main.tex:437–440): substantive

The sentence says simulated populations "recover broad contrasts while showing less heterogeneity than the people they stand in for, in markets, cognitive phenotyping, survey populations and persona-based simulation", with five citations. The sources split as follows.

- `delRioChanonaMarkets2025`: supports it ("LLMs exhibit less heterogeneity in behavior than humans").
- `suhSurveyDistributions2025` (Cao et al.): supports it ("all LLMs we tested, whether fine-tuned or not, are less diverse in their predictions across countries than the actual human survey data").
- `wangHumanSubjectivity2024` (Giorgi et al.): partial. Lack of variation is their motivating hypothesis. Their result is that "explicit LLM personas show mixed results when reproducing known human biases, but generally fail to demonstrate implicit biases".
- `codaFornoCogBench2024`: **does not support it.** It phenotypes 40 LLMs on ten behavioural metrics, and neither its abstract nor its framing reports reduced heterogeneity relative to humans.
- `wangSocioBench2025`: **does not support it.** It reports "only 30–40% accuracy when simulating individuals", which is an accuracy gap, not a heterogeneity gap.

The paper already cites `synthetic_replacement_human`, which is the cleanest direct support for this claim. Suggested replacement:

```latex
Simulated populations can recover broad contrasts while showing less
heterogeneity than the people they stand in for, in laboratory markets
\citep{delRioChanonaMarkets2025} and survey populations
\citep{synthetic_replacement_human,suhSurveyDistributions2025}, and persona
prompting does not restore the missing human factors
\citep{wangHumanSubjectivity2024}; broader benchmarks of cognitive phenotyping
and individual-level survey simulation report further gaps from human behaviour
\citep{codaFornoCogBench2024,wangSocioBench2025},
```

### 4.2 Pal et al. cited for "self-play need not predict cross-play" (main.tex:472–474): substantive

Lekeas supports this claim. Pal et al. do not. Their only direct LLM-versus-LLM games (Section "Repeated games over 10 rounds") report the same behaviour across all 15 pairings: "In the first treatment, we observe full cooperation in all games ... In the second treatment, all LLMs except GPT-5 cooperate in every round of every game". Their Nash and partner/rival analysis is about payoffs against random memory-1 opponents, not about behaviour differing between self-play and cross-play. Fix: cite only `lekeasNashSuppression2026` here and keep Pal at main.tex:448.

```latex
not predict its behaviour against a different agent is known
\citep{lekeasNashSuppression2026}, and fixed opponents are
standard practice in this literature
\citep{playing_repeated_games_with_llms,payneStrategicIntelligence2025};
```

(Adding Akata et al. also backs the phrase "standard practice" with more than one paper. Source main.tex:146 in Akata et al.: "we introduce three simplistic strategies ... always cooperate or defect ... an agent who defects in the first round but cooperates in all of the following rounds.")

### 4.3 "Public de-identified dataset" (main.tex:909–910; supplementary.tex:1242, 1257, 1896): substantive, unconfirmed

The source's arXiv v1 Data Availability statement (paper.tex:328) says: "All data will be publicly deposited in the same OSF repository of the pre-registration. A private link will be made available for reviewers". The OSF preregistration `pzyfm` is public. The OSF project it was registered from (`qcb5j`) returns "Authentication credentials were not provided", so the data could not be confirmed as public. Until a public URL exists, write "the de-identified dataset released by the authors of \citet{falling_behind_unsafe}". If the deposit is public, cite its URL or DOI. A reviewer who tries to find a "public" dataset and cannot will doubt the human comparison.

### 4.4 Minor claim-precision items

- **Strategy name (main.tex:570).** The paper writes "Conditional Unsafe (CAS)". The source names CAS "Conditionally Antisocial Safe" (paper.tex:311: "plays U in round 1, then from round 2 copies the opponent's previous action"). The behaviour matches, but the abbreviation no longer expands to the name used. Either use the source name, or write "Conditional Unsafe (CAS in \citet{falling_behind_unsafe})".
- **Payne rivals (supplementary.tex:1371–1372).** The text says Payne "run Tit-for-Tat and Grim Trigger", then "Our two conditional rivals are theirs." The CAS rival is Payne's *Suspicious* Tit-for-Tat (Body.tex:194: "begins by defecting on the first move ... After the first move, it behaves exactly like Tit for Tat"), not Grim Trigger. Suggested wording: "run Tit-for-Tat, Suspicious Tit-for-Tat and Grim Trigger ... Our two conditional rivals are their Tit-for-Tat and Suspicious Tit-for-Tat."
- **Akata fixed-opponent sentence (supplementary.tex:1369–1370).** Supported: five LLMs (GPT-4, text-davinci-003, text-davinci-002, Claude 2, Llama 2) against always-cooperate, always-defect and defect-once agents.
- **Tournament literature (supplementary.tex:1749).** Supported: Nieken and Sliwka (risk-taking depends on "the size of a potential lead"), Ozbeklik and Smith (lagging players take more risk), Gürtler et al. ("relative standing induces agents to take higher risks").

## 5. Uncited, duplicate and missing entries

**Missing keys:** none. Every `\cite` key in both files has a bib entry. (Note: the local `paper/main.bbl` and `main.aux` date from 2026-09-10 and lack `payneStrategicIntelligence2025` and `lekeasNashSuppression2026`. The PDF built on 2026-09-14 does contain both, so only the local build files are out of date.)

**Duplicate:** `ai_race_for_strategic_advantage` and `CaveOHeigeartaigh2018AIRace` are the same work (DOI 10.1145/3278721.3278780). Only the first is cited. Delete `CaveOHeigeartaigh2018AIRace`.

**Used but not cited: EGTtools.** The software is named at main.tex:1027 and supplementary.tex:737–761, and the analysis pins egttools 0.1.14.2 (`scripts/validate_egttools_pinned_source.py`), but `fernandezDomingos2020Egttools` is never cited. DataCite: title "EGTtools: Toolbox for Evolutionary Game Theory", creator Fernández Domingos, Elias. Concept DOI 10.5281/zenodo.3687125 currently resolves to release v0.1.14-patch2, with publicationYear 2025. The bib's year 2020 is not confirmed for this DOI. Suggested entry, cited at main.tex:1027:

```bibtex
@misc{fernandezDomingos2020Egttools,
  author       = {Fern{\'a}ndez Domingos, Elias},
  title        = {{EGTtools}: Toolbox for Evolutionary Game Theory},
  year         = {2025},
  version      = {v0.1.14-patch2},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.3687125},
  url          = {https://github.com/Socrats/EGTTools},
  note         = {Software}
}
```

**Never cited (17).** BibTeX prints nothing for these, so they are harmless in the PDF. Delete them or leave them.

`hendrycks2023overview`, `saeri2026prioritization`, `bengio2024managing`, `muller2026evolvable`, `han2019modelling`, `CaveOHeigeartaigh2018AIRace` (duplicate), `gruetzemacher2025strategic`, `ord2020precipice`, `han2022voluntary`, `minhEtAlValiditySurvey` (unverifiable), `mousaviDavoudiEtAl2026`, `liEtAl2026Fast`, `fernandezDomingos2020Egttools` (should be cited, see above), `shiEtAl2025Route`, `makelovEtAl2025Principled`, `aradEtAl2025Steering`, `piattiGovSim2024`.

**Supplement only (3):** `risk-taking_touraments`, `risk_taking_in_competition`, `competition_and_risk-taking`.

**Changelog out of date.** `paper/CITATION_CHANGELOG.md` says RouteSAE, Makelov and Arad are cited at main.tex:110 and 249, and it refers to `paper/refs.bib`. None of the three is cited any longer, and the file is `references.bib`. The changelog should be updated or marked historical so that it does not mislead the next audit.
