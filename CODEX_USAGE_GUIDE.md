# Codex Usage Guide

## Problem: Codex TUI Not Rendering

The TUI (Text User Interface) requires a minimum terminal size. If your terminal is too small, codex will hang without displaying anything.

### Solution 1: Resize Terminal

- **Minimum requirements**: 20+ rows, 80+ columns
- **Check current size**: `stty size`
- **Fix**: Maximize your terminal window or resize it larger

### Solution 2: Use Web Interface (Recommended)

#### Local Usage:
```bash
codex web
```
Then open: `http://localhost:4096/`

#### SSH Usage from Laptop:
```bash
# On your laptop, connect with port forwarding:
ssh -L 4096:localhost:4096 bs_thesis@172.28.137.17

# Then on your laptop's browser:
# Open: http://localhost:4096/
```

This forwards port 4096 from the remote machine to your laptop, allowing you to use the web interface as if it were running locally.

### Solution 3: Use in Full Terminal Emulator

Instead of VS Code's integrated terminal, use a full terminal:
```bash
# Open a standalone terminal (gnome-terminal, konsole, etc.)
codex
```

### Solution 4: Use with tmux

tmux provides better terminal management:
```bash
# Install tmux if needed
sudo apt install tmux  # or your package manager

# Start tmux
tmux

# Run codex
codex

# Detach from tmux: Ctrl+b then d
# Reattach: tmux attach
```

## Quick Reference

| Method | Command | When to Use |
|--------|---------|-------------|
| TUI | `codex` | Large terminal, local use |
| Web | `codex web` | Any browser access |
| SSH Web | `ssh -L 4096:localhost:4096 user@host` | Remote access from laptop |
| Attach | `codex attach URL` | Connect to running server |

## Authentication

- **Login**: `codex auth login`
- **Logout**: `codex auth logout`
- **Check status**: `codex auth list`

## Current Setup

- Version: 1.1.31
- Installation: `/home/bs_thesis/.local/bin/codex`
- Authenticated: OpenAI (OAuth)
