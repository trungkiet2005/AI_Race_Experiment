# STATE

## Now
Behaviour paper retargeted from AAMAS 2027 to IJCAI 2027 main track (user, 2026-11-22).
Sources in `paper/` are still the AAMAS sigconf build (8 body pages, last commit 2026-09-27).

## Lanes and owners
- Mac Claude: repo moved onto the Mac 2026-11-22 (clone + gitignored files copied from the Windows laptop over SSH).

## Running jobs
None.

## Next steps
1. Confirm whether the paper was submitted to AAMAS 2027 and is still under review; if so, withdraw it before 2027-01-11 (IJCAI forbids dual submission).
2. Port `paper/main.tex` to the IJCAI 2027 style (ijcai.org/authors_kit): 8 to 7 body pages, references up to 2 extra pages, author-year citations per IJCAI style.
3. Port `paper/supplementary.tex` to a separate anonymised PDF (50 MB cap); nothing a claim depends on may live only there.
4. Update `scripts/build_publication.py` and the anonymity checks for the new template.
5. Register abstract on the IJCAI submission site by 2027-01-04 AoE; full paper 2027-01-11 AoE.

## Decisions ledger
- 2026-11-22: target venue for the behaviour paper is IJCAI 2027 (user: "now it should be ijcai, we will aim for ijcai").
- 2026-11-22: project lives in `~/PhD_Farming/active/AI_Race_Experiment` on the Mac.

## Manual steps for the user
- Say whether an AAMAS 2027 submission exists (submission ID) so it can be withdrawn if needed.
