# Current MOEX tradability universe V1

The production candidate universe combines historical canonical mapping with one bounded current
MOEX ISS TQBR board snapshot. Historical membership remains available to research and event data,
but is not sufficient evidence of current production eligibility.

Candidates fail closed when the current security row is absent, the board is wrong, the lot is
invalid, security status is unknown or disabled, or current market representation is absent. The
structural gate does not require a non-null `LAST`; transient quote validity remains the market
adapter's responsibility.

Held positions bypass candidate exclusion so inactive or missing instruments remain visible to the
market and risk layers. Existing incomplete-mark BUY rejection and defensive SELL policy are not
changed.

`AGRO` and `RAGR` remain distinct instrument identities. Verified issuer succession is represented
as explicit metadata and never silently rewrites historical `AGRO` records.

Use `python -m apps.cli.production_burnin universe-status` for a read-only live classification.
Burn-in observations created before this gate are preserved as epoch 1 evidence. Final post-fix
acceptance starts in `production-dry-run-burnin-v1-epoch-2` and still requires five valid PRIMARY
cycles across five distinct confirmed MOEX trading days.
