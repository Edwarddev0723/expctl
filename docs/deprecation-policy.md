# Deprecation Policy

- beta 期間避免更改 CLI command names、config top-level sections、validation strategy names、tracking backend names、adapter entrypoints
- 若必須調整，先提供 warning 與 migration note
- legacy config compat layer 屬過渡機制，不視為長期穩定 public interface
