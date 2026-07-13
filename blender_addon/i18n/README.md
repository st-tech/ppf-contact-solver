# Localization catalogs

This directory holds the addon's UI translations. Blender's
`bpy.app.translations` system swaps the English UI text for a translation
when the user turns on **Preferences > Interface > Translation** and picks a
language. The loader in `__init__.py` reads every catalog here and registers
it when the addon starts.

## Files

- `en.json` is the master list of every translatable English string. It is an
  identity map (each key maps to itself) and is generated from the source
  code, so do not edit it by hand.
- `<locale>.json` (for example `ja_JP.json`, `zh_HANS.json`, `ko_KR.json`)
  maps each English string from `en.json` to its translation.

## Translating

You only edit the value side of a `<locale>.json` file. Nothing else in the
addon needs to change.

```json
{
  "Contact Solver": "コンタクトソルバー",
  "Connect": "接続",
  "Created pin '{name}' with {count} points": "{count} 点でピン '{name}' を作成しました"
}
```

Rules:

- An empty value (`""`) means "not translated yet". The UI shows the English
  source string for those, so a partial catalog is fine and safe to ship.
- Keep every `{placeholder}` token exactly as written, including the braces
  and any `:format` spec such as `{value:.2f}`. These are filled in at runtime.
  You may move a placeholder to wherever it reads naturally in your language.
  A translation whose placeholders do not match the source is ignored (the UI
  falls back to English) so a typo can never break the addon.
- Leave technical names untranslated: MCP, UUID, SSH, TCP, PC2, Docker,
  Jupyter, GPU, CUDA, Zozo.
- Preserve surrounding punctuation, quotes, ellipses, and trailing colons.

## Adding a language

1. Copy `en.json` to `<locale>.json`, where `<locale>` is a Blender language
   code (see the list in Preferences > Interface > Translation, for example
   `fr_FR`, `de_DE`, `es`, `zh_HANT`).
2. Replace each value with your translation, or leave it empty to defer.
3. That is all. The loader picks the file up automatically on the next start.
   No Python change and no code review of the strings is needed.
