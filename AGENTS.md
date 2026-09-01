# ClickUi managed project

## Repository
- GitHub: https://github.com/CodeUpdaterBot/ClickUi
- Managed path: `C:/Users/PC/Documents/Coding Projects/ClickUi`
- Default branch: `main`
- Visibility: public
- Product website source lives separately in `CodeUpdaterBot/ClickUi-Website`.

## Stack
Cross-platform Python desktop AI assistant using PySide6, speech recognition/TTS, model-provider APIs, and browser automation.

## Setup and verification
Windows uses the repository's Conda and pip dependency files. See `README.md` and `Install.bat` for the full environment setup.

Minimum source validation:

```bash
python -m py_compile clickui.py sonos.py
```

Run after dependencies and local configuration are installed:

```bash
python clickui.py
```

## Configuration and safety
- `.voiceconfig` is tracked as the repository's template/default configuration. Never commit real API keys, local browser-profile paths, or private conversation history.
- Keep runtime `history/`, `.env*`, caches, and logs untracked.
- Do not silently exercise paid model APIs or external account actions during verification.

## Release workflow
1. Pull `origin/main` before editing.
2. Make changes in this managed clone, not an older Cursor/download folder.
3. Run Python compilation and any relevant manual feature checks.
4. Review the diff and secret scan.
5. Commit as `Steven <runcomps@gmail.com>` and push to `main` when the user authorizes release.
