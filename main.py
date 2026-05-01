# ============================================================
# MAIN SCRIPT — Web UI version
# Opens the NavSafe dashboard in your browser.
# Upload images (and optional audio) through the UI.
# The real pipeline runs on the server and streams logs live.
# ============================================================

import sys
import os
import threading
import webbrowser
import json
import queue
import traceback
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, Response, send_from_directory

from config import (
    INPUT_IMAGES_DIR,
    INPUT_AUDIO_DIR,
    RESULTS_CSV_PATH,
    OUTPUTS_DIR,
    PREDICTED_MASKS_DIR,
    DEMO_IMAGES_DIR,
)

from src.pipeline import predict_folder
from src.route_recommendation import run_route_recommendation

import pandas as pd

# ============================================================
# Flask app
# ============================================================

app = Flask(__name__, static_folder=None)

# Global log queue — pipeline writes here, SSE reads from here
_log_queue: queue.Queue = queue.Queue()
_pipeline_running = False


# ============================================================
# Logging helper — replaces print() inside the pipeline
# ============================================================

class QueueLogger:
    """Redirect stdout to the SSE queue AND the real terminal."""

    def __init__(self, q: queue.Queue, original_stdout):
        self._q = q
        self._out = original_stdout
        self._buf = ""

    def write(self, text):
        self._out.write(text)
        self._out.flush()
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put({"type": "log", "text": line})

    def flush(self):
        self._out.flush()


# ============================================================
# Routes — static UI
# ============================================================

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NavSafe — Intelligent Route Safety Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --teal-950: #021b18;
    --teal-900: #042e28;
    --teal-800: #064d42;
    --teal-700: #097060;
    --teal-600: #0c9278;
    --teal-500: #10b896;
    --teal-400: #34d1ae;
    --teal-300: #6ee7d0;
    --teal-200: #a7f3e3;
    --teal-100: #d1faf2;
    --teal-50:  #edfdf9;
    --gray-950: #080c0b;
    --gray-900: #111614;
    --gray-800: #1a201e;
    --gray-700: #253029;  
    --gray-600: #374740;
    --gray-500: #546059;
    --gray-400: #7a8c85;
    --gray-300: #a3b5ae;
    --gray-200: #c8d8d2;
    --gray-100: #e3eeea;
    --gray-50:  #f2f8f5;
    --white:    #ffffff;
    --green-600: #166534; --green-100: #dcfce7; --green-500: #22c55e;
    --amber-600: #92400e; --amber-100: #fef3c7; --amber-500: #f59e0b;
    --red-600: #991b1b;   --red-100: #fee2e2;   --red-500: #ef4444;
    --sidebar-w: 230px; --header-h: 64px;
    --font: 'DM Sans', sans-serif;
    --display: 'Syne', sans-serif;
    --mono: 'IBM Plex Mono', monospace;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:var(--font);background:var(--gray-50);color:var(--gray-900);font-size:14px;line-height:1.6;display:flex;min-height:100vh;}

  /* SIDEBAR */
  .sidebar{width:var(--sidebar-w);min-height:100vh;background:var(--teal-950);display:flex;flex-direction:column;position:fixed;top:0;left:0;z-index:100;}
  .sidebar-logo{height:var(--header-h);display:flex;align-items:center;gap:12px;padding:0 22px;border-bottom:1px solid rgba(255,255,255,0.07);}
  .logo-mark{width:34px;height:34px;background:var(--teal-500);border-radius:9px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
  .logo-text{font-family:var(--display);font-size:17px;font-weight:700;color:#fff;letter-spacing:-0.01em;}
  .logo-text span{color:var(--teal-400);}
  .sidebar-section-label{font-size:10px;font-weight:500;letter-spacing:0.12em;text-transform:uppercase;color:var(--teal-700);padding:22px 22px 8px;}
  .nav-item{display:flex;align-items:center;gap:10px;padding:10px 22px;font-size:13px;font-weight:400;color:var(--teal-300);cursor:pointer;transition:all 0.15s;text-decoration:none;border:none;background:none;width:100%;text-align:left;position:relative;font-family:var(--font);}
  .nav-item:hover{color:#fff;background:rgba(255,255,255,0.05);}
  .nav-item.active{color:#fff;background:rgba(16,184,150,0.15);}
  .nav-item.active::before{content:'';position:absolute;left:0;top:5px;bottom:5px;width:3px;background:var(--teal-400);border-radius:0 3px 3px 0;}
  .nav-icon{width:16px;height:16px;opacity:0.6;flex-shrink:0;}
  .nav-item.active .nav-icon{opacity:1;}
  .sidebar-bottom{margin-top:auto;padding:18px 22px;border-top:1px solid rgba(255,255,255,0.06);}
  .user-row{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--teal-400);}
  .user-avatar{width:30px;height:30px;border-radius:50%;background:var(--teal-800);border:1px solid var(--teal-600);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:var(--teal-300);flex-shrink:0;font-family:var(--mono);}
  .user-name{color:var(--gray-200);font-weight:500;font-size:13px;}

  /* MAIN */
  .main{margin-left:var(--sidebar-w);flex:1;display:flex;flex-direction:column;min-height:100vh;}
  .topbar{height:var(--header-h);background:var(--white);border-bottom:1px solid var(--gray-100);display:flex;align-items:center;justify-content:space-between;padding:0 36px;position:sticky;top:0;z-index:50;box-shadow:var(--shadow-sm);}
  .breadcrumb{font-size:13px;color:var(--gray-400);font-family:var(--font);}
  .breadcrumb strong{color:var(--gray-900);font-weight:500;}
  .topbar-right{display:flex;align-items:center;gap:10px;}
  .status-pill{display:flex;align-items:center;gap:6px;background:var(--teal-50);border:1px solid var(--teal-100);border-radius:99px;padding:5px 14px;font-size:12px;color:var(--teal-700);}
  .status-dot{width:7px;height:7px;border-radius:50%;background:var(--teal-500);animation:pulse-dot 2s ease-in-out infinite;}
  @keyframes pulse-dot{0%,100%{opacity:1;}50%{opacity:0.3;}}
  .icon-btn{width:36px;height:36px;border-radius:8px;border:1px solid var(--gray-100);background:var(--white);display:flex;align-items:center;justify-content:center;cursor:pointer;color:var(--gray-400);transition:all 0.15s;}
  .icon-btn:hover{background:var(--gray-50);border-color:var(--gray-200);color:var(--gray-700);}

  /* PAGE */
  .page{padding:36px;flex:1;}
  .page-title{font-family:var(--display);font-size:24px;font-weight:700;color:var(--gray-950);letter-spacing:-0.03em;margin-bottom:4px;}
  .page-subtitle{font-size:13px;color:var(--gray-400);margin-bottom:32px;}

  /* STEP TRACK */
  .step-track{display:flex;align-items:center;margin-bottom:32px;background:var(--white);border:1px solid var(--gray-100);border-radius:12px;padding:18px 28px;box-shadow:var(--shadow-sm);}
  .step-node{display:flex;align-items:center;gap:10px;flex:1;}
  .step-node:not(:last-child)::after{content:'';flex:1;height:1px;background:var(--gray-100);margin:0 14px;}
  .step-circle{width:32px;height:32px;border-radius:50%;border:2px solid var(--gray-200);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:var(--gray-400);flex-shrink:0;transition:all 0.25s;background:var(--white);font-family:var(--mono);}
  .step-circle.active{border-color:var(--teal-500);background:var(--teal-500);color:#fff;box-shadow:0 0 0 4px rgba(16,184,150,0.15);}
  .step-circle.done{border-color:var(--green-500);background:var(--green-500);color:#fff;}
  .step-label{font-size:12px;font-weight:500;color:var(--gray-700);font-family:var(--display);}
  .step-desc{font-size:11px;color:var(--gray-400);}
  .step-node.active .step-label{color:var(--teal-700);}
  .step-node.done .step-label{color:var(--green-600);}

  /* CARDS */
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:22px;}
  .three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px;margin-bottom:22px;}
  .full-col{margin-bottom:22px;}
  .section-card{background:var(--white);border:1px solid var(--gray-100);border-radius:12px;box-shadow:var(--shadow-sm);overflow:hidden;}
  .section-card-header{display:flex;align-items:center;justify-content:space-between;padding:16px 22px;border-bottom:1px solid var(--gray-100);background:var(--white);}
  .section-card-title{font-size:13px;font-weight:600;color:var(--gray-800);display:flex;align-items:center;gap:8px;font-family:var(--display);}
  .section-card-title .icon{width:18px;height:18px;color:var(--teal-500);}
  .section-card-body{padding:22px;}

  /* PREFERENCES PANEL */
  .prefs-panel{background:var(--white);border:1px solid var(--teal-200);border-radius:12px;box-shadow:var(--shadow-md);overflow:hidden;margin-bottom:22px;}
  .prefs-panel-header{background:linear-gradient(135deg, var(--teal-900) 0%, var(--teal-950) 100%);padding:20px 24px;display:flex;align-items:center;gap:14px;}
  .prefs-header-icon{width:42px;height:42px;background:rgba(16,184,150,0.2);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;}
  .prefs-header-text{}
  .prefs-header-title{font-family:var(--display);font-size:15px;font-weight:700;color:#fff;letter-spacing:-0.01em;}
  .prefs-header-sub{font-size:12px;color:var(--teal-300);margin-top:2px;}
  .prefs-body{padding:24px;display:grid;grid-template-columns:1fr 1fr;gap:24px;}
  .pref-group{display:flex;flex-direction:column;gap:10px;}
  .pref-label{font-size:12px;font-weight:600;color:var(--gray-700);text-transform:uppercase;letter-spacing:0.07em;font-family:var(--display);}
  .pref-desc{font-size:12px;color:var(--gray-400);line-height:1.5;margin-top:-4px;}
  .pref-slider-wrap{display:flex;flex-direction:column;gap:6px;}
  .pref-value-display{font-family:var(--mono);font-size:22px;font-weight:500;color:var(--teal-700);letter-spacing:-0.02em;}
  .pref-value-display span{font-size:14px;color:var(--teal-500);}
  .slider{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:99px;background:var(--gray-100);outline:none;cursor:pointer;transition:background 0.2s;}
  .slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:20px;height:20px;border-radius:50%;background:var(--teal-500);cursor:pointer;border:3px solid var(--white);box-shadow:0 0 0 2px var(--teal-400), 0 2px 6px rgba(0,0,0,0.15);transition:all 0.15s;}
  .slider::-webkit-slider-thumb:hover{background:var(--teal-600);transform:scale(1.1);}
  .slider::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:var(--teal-500);cursor:pointer;border:3px solid var(--white);box-shadow:0 0 0 2px var(--teal-400);}
  .pref-range-labels{display:flex;justify-content:space-between;font-size:10px;color:var(--gray-300);font-family:var(--mono);}

  /* DROP ZONE */
  .drop-zone{border:1.5px dashed var(--teal-200);border-radius:10px;background:var(--teal-50);padding:28px 20px;text-align:center;cursor:pointer;transition:all 0.18s;}
  .drop-zone:hover,.drop-zone.drag-over{background:var(--teal-100);border-color:var(--teal-400);}
  .drop-zone input{display:none;}
  .drop-zone-icon{width:42px;height:42px;background:var(--teal-100);border-radius:10px;display:flex;align-items:center;justify-content:center;margin:0 auto 10px;color:var(--teal-600);}
  .drop-zone p{font-size:13px;font-weight:500;color:var(--teal-900);}
  .drop-zone span{font-size:11px;color:var(--gray-400);display:block;margin-top:3px;}

  /* FILE LIST */
  .file-list{display:flex;flex-direction:column;gap:6px;margin-top:12px;}
  .file-item{display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--white);border:1px solid var(--gray-100);border-radius:8px;font-size:12px;}
  .file-thumb{width:34px;height:34px;border-radius:6px;background:var(--teal-100);display:flex;align-items:center;justify-content:center;color:var(--teal-600);font-size:13px;flex-shrink:0;overflow:hidden;}
  .file-thumb img{width:100%;height:100%;object-fit:cover;border-radius:6px;}
  .file-meta{flex:1;}
  .file-name{font-weight:500;color:var(--gray-800);font-size:12px;}
  .file-size{font-size:11px;color:var(--gray-400);}
  .file-remove{color:var(--gray-300);cursor:pointer;font-size:18px;line-height:1;transition:color 0.1s;}
  .file-remove:hover{color:var(--red-500);}

  /* RUN BUTTON */
  .run-btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;padding:14px;background:var(--teal-600);color:#fff;border:none;border-radius:10px;font-size:14px;font-weight:600;font-family:var(--display);cursor:pointer;transition:all 0.18s;box-shadow:0 4px 14px rgba(12,146,120,0.35);letter-spacing:-0.01em;}
  .run-btn:hover{background:var(--teal-700);box-shadow:0 6px 20px rgba(12,146,120,0.4);transform:translateY(-1px);}
  .run-btn:active{transform:translateY(0);}
  .run-btn:disabled{background:var(--gray-200);color:var(--gray-400);box-shadow:none;cursor:not-allowed;transform:none;}

  /* CONSOLE */
  .console{background:var(--gray-950);border-radius:10px;font-family:var(--mono);font-size:12px;line-height:1.85;padding:16px;min-height:140px;max-height:260px;overflow-y:auto;}
  .console::-webkit-scrollbar{width:4px;}
  .console::-webkit-scrollbar-thumb{background:var(--gray-700);border-radius:4px;}
  .log-line{display:block;color:#6b7280;}
  .log-line.ok{color:#34d399;}
  .log-line.warn{color:#fbbf24;}
  .log-line.err{color:#f87171;}
  .log-line.info{color:var(--teal-400);}
  .log-line.head{color:#e5e7eb;font-weight:500;}

  /* METRICS */
  .metrics-row{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:22px;}
  .metric-tile{background:var(--white);border:1px solid var(--gray-100);border-radius:12px;padding:18px;box-shadow:var(--shadow-sm);}
  .metric-tile-label{font-size:10px;color:var(--gray-400);font-weight:500;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;font-family:var(--display);}
  .metric-tile-value{font-size:26px;font-weight:700;color:var(--gray-950);letter-spacing:-0.04em;line-height:1;font-family:var(--display);}
  .metric-tile-sub{font-size:11px;color:var(--gray-400);margin-top:4px;}
  .metric-tile.accent{border-color:var(--teal-200);background:var(--teal-50);}
  .metric-tile.accent .metric-tile-value{color:var(--teal-700);}

  /* TABLE */
  .data-table{width:100%;border-collapse:collapse;}
  .data-table thead tr{border-bottom:1px solid var(--gray-100);}
  .data-table th{font-size:10px;font-weight:600;color:var(--gray-400);text-transform:uppercase;letter-spacing:0.08em;padding:11px 16px;text-align:left;white-space:nowrap;font-family:var(--display);}
  .data-table td{padding:12px 16px;font-size:12px;color:var(--gray-800);border-bottom:1px solid var(--gray-50);white-space:nowrap;}
  .data-table tbody tr:hover{background:var(--gray-50);}
  .data-table tbody tr:last-child td{border-bottom:none;}

  /* BADGES */
  .badge{display:inline-flex;align-items:center;padding:3px 9px;border-radius:99px;font-size:11px;font-weight:600;font-family:var(--mono);}
  .badge-low{background:var(--green-100);color:var(--green-600);}
  .badge-medium{background:var(--amber-100);color:var(--amber-600);}
  .badge-high{background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;}
  .badge-critical{background:var(--red-100);color:var(--red-600);border:1px solid #fca5a5;}
  .badge-teal{background:var(--teal-100);color:var(--teal-700);}

  /* BAR */
  .bar-row{display:flex;align-items:center;gap:8px;}
  .bar-bg{flex:1;height:4px;background:var(--gray-100);border-radius:99px;overflow:hidden;min-width:60px;}
  .bar-fill{height:100%;border-radius:99px;transition:width 0.6s ease;}
  .bar-val{font-size:11px;color:var(--gray-500);min-width:44px;font-family:var(--mono);}

  /* ROUTE HERO */
  .route-hero{background:linear-gradient(135deg,var(--teal-800) 0%,var(--teal-950) 100%);border-radius:12px;padding:26px 30px;display:flex;align-items:center;gap:22px;margin-bottom:22px;position:relative;overflow:hidden;}
  .route-hero::before{content:'';position:absolute;top:-50px;right:-50px;width:180px;height:180px;border-radius:50%;background:rgba(255,255,255,0.03);}
  .route-hero::after{content:'';position:absolute;bottom:-30px;right:80px;width:100px;height:100px;border-radius:50%;background:rgba(16,184,150,0.07);}
  .route-hero-icon{width:54px;height:54px;border-radius:14px;background:rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0;position:relative;z-index:1;}
  .route-hero-text{position:relative;z-index:1;}
  .route-hero-label{font-size:10px;color:var(--teal-400);font-weight:500;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;font-family:var(--display);}
  .route-hero-name{font-family:var(--display);font-size:22px;font-weight:700;color:#fff;letter-spacing:-0.03em;}
  .route-hero-sub{font-size:12px;color:var(--teal-300);margin-top:3px;}

  /* NOTIFICATION */
  .notification-panel{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:16px 20px;display:flex;gap:12px;align-items:flex-start;margin-bottom:22px;}
  .notification-text{font-size:13px;color:var(--amber-600);line-height:1.65;}
  .notification-text strong{color:var(--amber-600);font-weight:600;}

  /* OUTPUT FILES */
  .outputs-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
  .output-item{display:flex;align-items:center;gap:12px;padding:13px 16px;background:var(--white);border:1px solid var(--gray-100);border-radius:8px;}
  .output-icon{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;}
  .output-icon.csv{background:var(--green-100);}
  .output-icon.dir{background:var(--teal-100);}
  .output-name{font-size:12px;font-weight:500;color:var(--gray-800);font-family:var(--mono);}
  .output-desc{font-size:11px;color:var(--gray-400);margin-top:1px;}
  .output-check{margin-left:auto;color:var(--green-500);font-size:14px;font-weight:700;}

  /* MISC */
  .hidden{display:none!important;}
  .spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite;flex-shrink:0;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .divider{border:none;border-top:1px solid var(--gray-100);margin:28px 0;}
  .tag{display:inline-flex;align-items:center;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:500;background:var(--teal-50);color:var(--teal-700);border:1px solid var(--teal-100);font-family:var(--mono);}
  .error-banner{background:#fee2e2;border:1px solid #fca5a5;border-radius:10px;padding:16px 20px;display:flex;gap:12px;align-items:flex-start;margin-bottom:22px;}
  .error-banner-text{font-size:13px;color:var(--red-600);line-height:1.65;}
  .section-title{font-family:var(--display);font-size:17px;font-weight:700;color:var(--gray-900);letter-spacing:-0.025em;margin-bottom:4px;}
  .section-sub{font-size:13px;color:var(--gray-400);margin-bottom:22px;}
  .badge-count{display:inline-flex;align-items:center;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;background:var(--teal-100);color:var(--teal-700);font-family:var(--mono);}
</style>
</head>
<body>

<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-mark">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="18" height="18">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <span class="logo-text">Nav<span>Safe</span></span>
  </div>
  <div class="sidebar-section-label">Workspace</div>
  <button class="nav-item active">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
    Pipeline
  </button>
  <button class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/></svg>
    Results
  </button>
  <button class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    Analytics
  </button>
  <button class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l19-9-9 19-2-8-8-2z"/></svg>
    Routes
  </button>
  <div class="sidebar-section-label">System</div>
  <button class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93l-1.41 1.41M5.34 18.66l-1.41 1.41M2 12h2M20 12h2M19.07 19.07l-1.41-1.41M5.34 5.34L3.93 3.93M12 2v2M12 20v2"/></svg>
    Settings
  </button>
  <div class="sidebar-bottom">
    <div class="user-row">
      <div class="user-avatar">NS</div>
      <div>
        <div class="user-name">NavSafe Admin</div>
        <div style="font-size:11px;">v2.4.1 · production</div>
      </div>
    </div>
  </div>
</aside>

<main class="main">
  <header class="topbar">
    <div>
      <span class="breadcrumb">Pipeline <span style="margin:0 6px;color:var(--gray-300)">/</span> <strong>New Analysis Run</strong></span>
    </div>
    <div class="topbar-right">
      <div class="status-pill"><div class="status-dot"></div>Models ready</div>
      <div class="icon-btn" title="Notifications">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
      </div>
    </div>
  </header>

  <div class="page">
    <div class="page-title">Road Safety Analysis</div>
    <div class="page-subtitle">Upload road images and optional audio files (matched by filename), set your route preferences, then run the full prediction and recommendation pipeline.</div>

    <!-- STEPS -->
    <div class="step-track">
      <div class="step-node active" id="sn-0"><div class="step-circle active" id="sc-0">1</div><div><div class="step-label">Upload inputs</div><div class="step-desc">Images + audio</div></div></div>
      <div class="step-node" id="sn-1"><div class="step-circle" id="sc-1">2</div><div><div class="step-label">Running pipeline</div><div class="step-desc">Prediction models</div></div></div>
      <div class="step-node" id="sn-2"><div class="step-circle" id="sc-2">3</div><div><div class="step-label">Image results</div><div class="step-desc">Danger scores</div></div></div>
      <div class="step-node" id="sn-3"><div class="step-circle" id="sc-3">4</div><div><div class="step-label">Route recommendation</div><div class="step-desc">Final decision</div></div></div>
    </div>

    <!-- USER PREFERENCES -->
    <div class="prefs-panel">
      <div class="prefs-panel-header">
        <div class="prefs-header-icon">⚙️</div>
        <div class="prefs-header-text">
          <div class="prefs-header-title">Route Preferences</div>
          <div class="prefs-header-sub">Configure the thresholds used by the route recommendation engine.</div>
        </div>
      </div>
      <div class="prefs-body">
        <div class="pref-group">
          <div class="pref-label">Minimum Risk Reduction</div>
          <div class="pref-desc">Minimum percentage of risk reduction required before switching to an alternative route.</div>
          <div class="pref-value-display" id="risk-display">10<span>%</span></div>
          <div class="pref-slider-wrap">
            <input type="range" class="slider" id="risk-slider" min="0" max="100" value="10" step="1" oninput="updatePref('risk')">
            <div class="pref-range-labels"><span>0%</span><span>50%</span><span>100%</span></div>
          </div>
        </div>
        <div class="pref-group">
          <div class="pref-label">Maximum Extra Travel Time</div>
          <div class="pref-desc">Maximum percentage of extra travel time the user is willing to accept for a safer route.</div>
          <div class="pref-value-display" id="time-display">20<span>%</span></div>
          <div class="pref-slider-wrap">
            <input type="range" class="slider" id="time-slider" min="0" max="200" value="20" step="5" oninput="updatePref('time')">
            <div class="pref-range-labels"><span>0%</span><span>100%</span><span>200%</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- INPUTS -->
    <div class="two-col">
      <div class="section-card">
        <div class="section-card-header">
          <div class="section-card-title">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Input Images
          </div>
          <span class="badge-count hidden" id="img-count-badge"></span>
        </div>
        <div class="section-card-body">
          <div class="drop-zone" id="img-drop" onclick="document.getElementById('img-input').click()">
            <input type="file" id="img-input" multiple accept=".jpg,.jpeg,.png,.bmp,.webp" onchange="addImages(this.files)">
            <div class="drop-zone-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            </div>
            <p>Drag & drop road images here</p>
            <span>.jpg .jpeg .png .bmp .webp</span>
          </div>
          <div class="file-list" id="img-list"></div>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:22px;">
        <div class="section-card" style="flex:1;">
          <div class="section-card-header">
            <div class="section-card-title">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
              Audio Files
            </div>
            <span style="font-size:11px;color:var(--gray-400);">Optional · matched by filename</span>
          </div>
          <div class="section-card-body">
            <div class="drop-zone" id="aud-drop" onclick="document.getElementById('aud-input').click()">
              <input type="file" id="aud-input" multiple accept=".wav,.mp3,.flac,.ogg,.m4a" onchange="addAudio(this.files)">
              <div class="drop-zone-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/></svg>
              </div>
              <p>Select audio files</p>
              <span>.wav .mp3 .flac .ogg .m4a — each audio is paired to the image with the same filename</span>
            </div>
            <div class="file-list" id="aud-list"></div>
          </div>
        </div>

        <div class="section-card">
          <div class="section-card-body" style="padding:18px;">
            <button class="run-btn" id="run-btn" onclick="runPipeline()" disabled>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              Run Pipeline
            </button>
            <div style="margin-top:12px;font-size:11px;color:var(--gray-400);line-height:1.75;">
              Runs <span class="tag">predict_folder()</span> → <span class="tag">run_route_recommendation()</span>. Outputs saved to <span class="tag">outputs/</span>.
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- CONSOLE -->
    <div class="section-card full-col">
      <div class="section-card-header">
        <div class="section-card-title">
          <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
          Console Output
        </div>
        <button onclick="clearLog()" style="font-size:11px;color:var(--gray-400);background:none;border:none;cursor:pointer;font-family:var(--font);">Clear</button>
      </div>
      <div class="section-card-body" style="padding:14px;">
        <div class="console" id="log-box"><span class="log-line">Waiting to run…</span></div>
      </div>
    </div>

    <!-- RESULTS -->
    <div id="results-section" class="hidden">
      <hr class="divider">
      <div class="section-title">Image Prediction Results</div>
      <div class="section-sub">Per-image danger scores, weather classification, and object detection counts.</div>

      <div class="metrics-row" id="metrics-row"></div>

      <div class="section-card full-col">
        <div class="section-card-header">
          <div class="section-card-title">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
            results.csv
          </div>
          <span class="tag" id="row-count-tag"></span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Image name</th>
                <th>Final danger score</th>
                <th>Mean risk score</th>
                <th>Danger level</th>
                <th>Visual risk</th>
                <th>Weather</th>
                <th>Weather score</th>
                <th>People</th>
                <th>Vehicles</th>
                <th>Bikes</th>
              </tr>
            </thead>
            <tbody id="results-body"></tbody>
          </table>
        </div>
      </div>

      <div id="route-section" class="hidden">
        <hr class="divider">
        <div class="section-title">Route Recommendation</div>
        <div class="section-sub">Aggregated route-level scores and the final recommended route based on your preferences.</div>

        <div class="route-hero">
          <div class="route-hero-icon">🗺️</div>
          <div class="route-hero-text">
            <div class="route-hero-label">Recommended route</div>
            <div class="route-hero-name" id="route-hero-name">—</div>
            <div class="route-hero-sub" id="route-hero-sub">—</div>
          </div>
        </div>

        <div class="notification-panel">
          <div style="font-size:16px;flex-shrink:0;">⚠️</div>
          <div class="notification-text" id="notification-text">—</div>
        </div>

        <div class="section-card full-col">
          <div class="section-card-header">
            <div class="section-card-title">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l19-9-9 19-2-8-8-2z"/></svg>
              route_summary.csv
            </div>
          </div>
          <div style="overflow-x:auto;">
            <table class="data-table">
              <thead>
                <tr><th>Route</th><th>Avg danger</th><th>Max danger</th><th>Travel time</th><th>Images</th><th>People</th><th>Vehicles</th><th>Danger level</th></tr>
              </thead>
              <tbody id="route-body"></tbody>
            </table>
          </div>
        </div>
      </div>

      <div id="route-error-section" class="hidden">
        <hr class="divider">
        <div class="error-banner">
          <div style="font-size:16px;flex-shrink:0;">⚠️</div>
          <div class="error-banner-text" id="route-error-text">Route recommendation failed.</div>
        </div>
      </div>

      <!-- OUTPUT FILES -->
      <hr class="divider">
      <div class="section-title">Output Files</div>
      <div class="section-sub">Saved to <span class="tag">outputs/</span></div>
      <div class="outputs-grid">
        <div class="output-item"><div class="output-icon csv">📄</div><div><div class="output-name">results.csv</div><div class="output-desc">Image-level predictions</div></div><div class="output-check">✓</div></div>
        <div class="output-item"><div class="output-icon dir">📁</div><div><div class="output-name">predicted_masks/</div><div class="output-desc">Segmentation masks</div></div><div class="output-check">✓</div></div>
        <div class="output-item"><div class="output-icon dir">📁</div><div><div class="output-name">demo_images/</div><div class="output-desc">Annotated demo frames</div></div><div class="output-check">✓</div></div>
        <div class="output-item"><div class="output-icon csv">📄</div><div><div class="output-name">route_summary.csv</div><div class="output-desc">Per-route aggregated scores</div></div><div class="output-check">✓</div></div>
        <div class="output-item"><div class="output-icon csv">📄</div><div><div class="output-name">route_decision.csv</div><div class="output-desc">Final route decision</div></div><div class="output-check">✓</div></div>
      </div>
    </div>

  </div><!-- /page -->
</main>

<script>
  const imageFiles = [];
  const audioFiles = [];  // multiple audio files, matched by stem

  // ── Preferences ─────────────────────────────────────────
  function updatePref(type) {
    if (type === 'risk') {
      const v = document.getElementById('risk-slider').value;
      document.getElementById('risk-display').innerHTML = v + '<span>%</span>';
    } else {
      const v = document.getElementById('time-slider').value;
      document.getElementById('time-display').innerHTML = v + '<span>%</span>';
    }
  }

  function getPrefs() {
    return {
      min_risk_reduction_pct: parseFloat(document.getElementById('risk-slider').value),
      max_extra_time_pct:     parseFloat(document.getElementById('time-slider').value),
    };
  }

  // ── File helpers ─────────────────────────────────────────
  function fmtSize(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
  }

  function stem(filename) {
    return filename.replace(/\.[^/.]+$/, '');
  }

  function addImages(files) {
    for (const f of files) {
      if (!imageFiles.find(x => x.name === f.name)) imageFiles.push(f);
    }
    renderImageList(); updateRunBtn();
  }

  function addAudio(files) {
    for (const f of files) {
      if (!audioFiles.find(x => x.name === f.name)) audioFiles.push(f);
    }
    renderAudioList();
  }

  function renderImageList() {
    const el = document.getElementById('img-list');
    const badge = document.getElementById('img-count-badge');
    if (!imageFiles.length) { el.innerHTML = ''; badge.classList.add('hidden'); return; }
    badge.classList.remove('hidden');
    badge.textContent = imageFiles.length + ' file' + (imageFiles.length > 1 ? 's' : '');
    const audioStems = new Set(audioFiles.map(a => stem(a.name)));
    el.innerHTML = imageFiles.map((f, i) => {
      const url = URL.createObjectURL(f);
      const hasPair = audioStems.has(stem(f.name));
      return `<div class="file-item">
        <div class="file-thumb"><img src="${url}" alt="${f.name}"></div>
        <div class="file-meta">
          <div class="file-name">${f.name}</div>
          <div class="file-size">${fmtSize(f.size)}${hasPair ? ' · <span style="color:var(--teal-600);font-weight:500;">audio paired ✓</span>' : ''}</div>
        </div>
        <span class="file-remove" onclick="removeImage(${i})">×</span>
      </div>`;
    }).join('');
  }

  function renderAudioList() {
    const el = document.getElementById('aud-list');
    if (!audioFiles.length) { el.innerHTML = ''; return; }
    el.innerHTML = audioFiles.map((f, i) => `
      <div class="file-item">
        <div class="file-thumb" style="background:var(--teal-100);color:var(--teal-600);">🎙</div>
        <div class="file-meta">
          <div class="file-name">${f.name}</div>
          <div class="file-size">${fmtSize(f.size)}</div>
        </div>
        <span class="file-remove" onclick="removeAudio(${i})">×</span>
      </div>`).join('');
    // Re-render image list to refresh pairing badges
    renderImageList();
  }

  function removeImage(i) { imageFiles.splice(i,1); renderImageList(); updateRunBtn(); }
  function removeAudio(i) { audioFiles.splice(i,1); renderAudioList(); }
  function updateRunBtn() { document.getElementById('run-btn').disabled = imageFiles.length === 0; }

  ['img-drop','aud-drop'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('dragover',  e => { e.preventDefault(); el.classList.add('drag-over'); });
    el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
    el.addEventListener('drop', e => {
      e.preventDefault(); el.classList.remove('drag-over');
      if (id === 'img-drop') addImages(e.dataTransfer.files);
      else addAudio(e.dataTransfer.files);
    });
  });

  // ── Console ───────────────────────────────────────────────
  function addLogLine(text, cls) {
    const box = document.getElementById('log-box');
    const line = document.createElement('span');
    line.className = 'log-line' + (cls ? ' ' + cls : '');
    line.textContent = text;
    box.appendChild(document.createElement('br'));
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }
  function clearLog() {
    document.getElementById('log-box').innerHTML = '<span class="log-line">Console cleared.</span>';
  }

  // ── Steps ─────────────────────────────────────────────────
  function setStep(n) {
    for (let i = 0; i < 4; i++) {
      const c = document.getElementById('sc-'+i);
      const s = document.getElementById('sn-'+i);
      if (i < n) {
        c.className = 'step-circle done';
        c.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
        s.className = 'step-node done';
      } else if (i === n) {
        c.className = 'step-circle active'; c.textContent = i+1;
        s.className = 'step-node active';
      } else {
        c.className = 'step-circle'; c.textContent = i+1;
        s.className = 'step-node';
      }
    }
  }

  // ── Render helpers ────────────────────────────────────────
  function dangerBadge(level) {
    if (!level || level === 'None' || level === 'null') return '—';
    const l = String(level).toLowerCase();
    const cls = l.includes('critical') ? 'badge-critical' : l.includes('high') ? 'badge-high' : l.includes('medium') ? 'badge-medium' : 'badge-low';
    return `<span class="badge ${cls}">${level}</span>`;
  }
  function bar(val, color) {
    if (val === null || val === undefined || isNaN(val)) return '—';
    const pct = Math.min(100, Math.round(parseFloat(val)*100));
    return `<div class="bar-row"><div class="bar-bg"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div><span class="bar-val">${parseFloat(val).toFixed(4)}</span></div>`;
  }
  function dangerColor(lvl) {
    const l = String(lvl||'').toLowerCase();
    return l.includes('critical') ? '#ef4444' : l.includes('high') ? '#f59e0b' : l.includes('medium') ? '#ca8a04' : '#22c55e';
  }
  function fmt(v, digits=4) {
    if (v === null || v === undefined || v === 'None' || v === '' || String(v) === 'nan') return '—';
    const n = parseFloat(v);
    return isNaN(n) ? String(v) : n.toFixed(digits);
  }

  // ── Run ───────────────────────────────────────────────────
  async function runPipeline() {
    const btn = document.getElementById('run-btn');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Running…';
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('route-section').classList.add('hidden');
    document.getElementById('route-error-section').classList.add('hidden');
    clearLog();
    setStep(1);

    const prefs = getPrefs();

    const formData = new FormData();
    imageFiles.forEach(f => formData.append('images', f));
    audioFiles.forEach(f => formData.append('audio_files', f));  // multiple audio files
    formData.append('min_risk_reduction_pct', prefs.min_risk_reduction_pct);
    formData.append('max_extra_time_pct', prefs.max_extra_time_pct);

    addLogLine('Uploading files to server…', 'info');
    addLogLine(`Preferences — min risk reduction: ${prefs.min_risk_reduction_pct}% · max extra time: ${prefs.max_extra_time_pct}%`, 'info');
    let runId;
    try {
      const res = await fetch('/api/run', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      runId = data.run_id;
    } catch(e) {
      addLogLine('Upload error: ' + e.message, 'err');
      btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Pipeline';
      btn.disabled = false;
      return;
    }

    // Stream logs via SSE
    setStep(1);
    const evtSource = new EventSource('/api/stream/' + runId);

    evtSource.addEventListener('log', e => {
      const d = JSON.parse(e.data);
      const text = d.text || '';
      const cls = text.includes('Error') || text.includes('error') || text.includes('FAILED') ? 'err'
                : text.includes('====') ? 'head'
                : text.startsWith('Loading') || text.includes('loaded') ? 'info'
                : text.includes('successfully') || text.includes('saved') || text.includes('✓') ? 'ok'
                : text.includes('Warning') || text.includes('warning') ? 'warn'
                : '';
      addLogLine(text, cls);
    });

    evtSource.addEventListener('step', e => {
      const d = JSON.parse(e.data);
      setStep(d.step);
    });

    evtSource.addEventListener('results', e => {
      const data = JSON.parse(e.data);
      renderResults(data);
    });

    evtSource.addEventListener('route', e => {
      const data = JSON.parse(e.data);
      renderRoute(data);
    });

    evtSource.addEventListener('route_error', e => {
      const d = JSON.parse(e.data);
      document.getElementById('route-error-text').textContent = 'Route recommendation failed: ' + d.error;
      document.getElementById('route-error-section').classList.remove('hidden');
    });

    evtSource.addEventListener('done', e => {
      evtSource.close();
      setStep(4);
      document.getElementById('results-section').classList.remove('hidden');
      btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Pipeline';
      btn.disabled = false;
    });

    evtSource.addEventListener('error_fatal', e => {
      const d = JSON.parse(e.data);
      addLogLine('Fatal error: ' + d.error, 'err');
      evtSource.close();
      btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Pipeline';
      btn.disabled = false;
    });
  }

  function renderResults(data) {
    const results = data.rows;
    setStep(2);

    const avg = results.reduce((s,r) => s + (parseFloat(r.final_danger_score)||0), 0) / results.length;
    const lvls = results.map(r => r.danger_level || '');
    const maxLvl = lvls.includes('Critical') ? 'Critical' : lvls.includes('High') ? 'High' : lvls.includes('Medium') ? 'Medium' : 'Low';

    document.getElementById('metrics-row').innerHTML = `
      <div class="metric-tile accent"><div class="metric-tile-label">Images processed</div><div class="metric-tile-value">${results.length}</div></div>
      <div class="metric-tile"><div class="metric-tile-label">Avg danger score</div><div class="metric-tile-value">${avg.toFixed(3)}</div><div class="metric-tile-sub">mean final_danger_score</div></div>
      <div class="metric-tile"><div class="metric-tile-label">Max danger level</div><div class="metric-tile-value" style="font-size:18px;margin-top:4px;">${dangerBadge(maxLvl)}</div></div>
      <div class="metric-tile"><div class="metric-tile-label">Total people</div><div class="metric-tile-value">${results.reduce((s,r)=>s+(parseInt(r.num_people)||0),0)}</div><div class="metric-tile-sub">across all images</div></div>
      <div class="metric-tile"><div class="metric-tile-label">Total vehicles</div><div class="metric-tile-value">${results.reduce((s,r)=>s+(parseInt(r.num_vehicles)||0),0)}</div><div class="metric-tile-sub">across all images</div></div>
    `;

    document.getElementById('row-count-tag').textContent = results.length + ' rows';
    document.getElementById('results-body').innerHTML = results.map(r => `
      <tr>
        <td style="font-family:var(--mono);font-size:11px;color:var(--gray-500)">${r.image_name}</td>
        <td>${bar(r.final_danger_score, dangerColor(r.danger_level))}</td>
        <td>${fmt(r.mean_risk_score)}</td>
        <td>${dangerBadge(r.danger_level)}</td>
        <td>${bar(r.visual_risk_score, '#818cf8')}</td>
        <td>${r.predicted_weather || '—'}</td>
        <td style="font-family:var(--mono);font-size:11px;">${fmt(r.weather_score)}</td>
        <td>${r.num_people ?? '—'}</td>
        <td>${r.num_vehicles ?? '—'}</td>
        <td>${r.num_bikes ?? '—'}</td>
      </tr>
    `).join('');
  }

  function renderRoute(data) {
    setStep(3);
    document.getElementById('route-hero-name').textContent = data.selected_route;
    document.getElementById('route-hero-sub').textContent =
      `Avg danger: ${parseFloat(data.avg_danger).toFixed(3)} · Travel time: ${data.travel_time} min · Level: ${data.danger_level}`;
    document.getElementById('notification-text').textContent = data.notification;

    document.getElementById('route-body').innerHTML = data.summary.map(r => `
      <tr style="${r.route_name === data.selected_route ? 'background:var(--teal-50);' : ''}">
        <td style="font-weight:${r.route_name === data.selected_route ? '600' : '400'};color:${r.route_name === data.selected_route ? 'var(--teal-700)' : 'var(--gray-800)'}">
          ${r.route_name === data.selected_route ? '★ ' : ''}${r.route_name}
        </td>
        <td>${bar(r.avg_danger_score, dangerColor(r.danger_level))}</td>
        <td style="font-family:var(--mono);font-size:11px;">${parseFloat(r.max_danger_score).toFixed(4)}</td>
        <td>${r.travel_time_min} min</td>
        <td>${r.num_images}</td>
        <td>${r.total_people}</td>
        <td>${r.total_vehicles}</td>
        <td>${dangerBadge(r.danger_level)}</td>
      </tr>
    `).join('');

    document.getElementById('route-section').classList.remove('hidden');
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ============================================================
# Routes — API
# ============================================================

# run_id → Queue of SSE events
_run_queues: dict = {}


@app.route("/api/run", methods=["POST"])
def api_run():
    global _pipeline_running

    if _pipeline_running:
        return jsonify({"error": "Pipeline is already running"}), 429

    # ── save uploaded images ─────────────────────────────────
    images = request.files.getlist("images")
    audio_files_uploaded = request.files.getlist("audio_files")  # multiple audio files

    if not images:
        return jsonify({"error": "No images uploaded"}), 400

    # Read user preferences from form data (with defaults)
    try:
        min_risk_reduction_pct = float(request.form.get("min_risk_reduction_pct", 10.0))
    except (TypeError, ValueError):
        min_risk_reduction_pct = 10.0

    try:
        max_extra_time_pct = float(request.form.get("max_extra_time_pct", 20.0))
    except (TypeError, ValueError):
        max_extra_time_pct = 20.0

    # Clear old input images
    for old in INPUT_IMAGES_DIR.iterdir():
        if old.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            old.unlink()

    image_paths = []
    for img in images:
        dest = INPUT_IMAGES_DIR / img.filename
        img.save(str(dest))
        image_paths.append(dest)

    # Save all uploaded audio files; build stem→path map for pairing
    for old in INPUT_AUDIO_DIR.iterdir():
        if old.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            old.unlink()

    audio_stem_map: dict = {}  # stem → Path
    for af in audio_files_uploaded:
        if af and af.filename:
            audio_dest = INPUT_AUDIO_DIR / af.filename
            af.save(str(audio_dest))
            audio_stem_map[Path(af.filename).stem] = audio_dest

    # ── create run queue & start background thread ───────────
    import uuid
    run_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    _run_queues[run_id] = q

    def pipeline_thread():
        global _pipeline_running
        _pipeline_running = True
        original_stdout = sys.stdout
        sys.stdout = QueueLogger(q, original_stdout)
        try:
            _run_pipeline(q, image_paths, audio_stem_map,
                          min_risk_reduction_pct, max_extra_time_pct)
        finally:
            sys.stdout = original_stdout
            _pipeline_running = False

    t = threading.Thread(target=pipeline_thread, daemon=True)
    t.start()

    return jsonify({"run_id": run_id})


def _run_pipeline(q: queue.Queue, image_paths, audio_stem_map: dict,
                  min_risk_reduction_pct: float, max_extra_time_pct: float):
    """Runs the real pipeline and pushes SSE events to q."""
    def event(typ, data):
        q.put({"type": typ, "data": data})

    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
        DEMO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        event("step", {"step": 1})

        print(f"\n================ PREFERENCES ================")
        print(f"Min risk reduction : {min_risk_reduction_pct}%")
        print(f"Max extra time     : {max_extra_time_pct}%")

        print("\n================ INPUT FILES ================")
        print("Input images found:")
        for image_path in sorted(image_paths):
            print("-", image_path.name)

        # ── IMAGE + AUDIO PAIRING (per-image, matched by filename stem) ──
        results = []

        for image_path in sorted(image_paths):
            # Match audio by same stem (e.g. road1.jpg ↔ road1.wav)
            audio_path = audio_stem_map.get(image_path.stem, None)

            if audio_path is not None:
                print(f"\nAudio found for {image_path.name}: {audio_path.name}")
            else:
                print(f"\nNo audio for {image_path.name}")

            result_df = predict_folder(
                image_paths=[image_path],
                audio_path=audio_path,
            )
            results.append(result_df)

        # Combine all per-image results
        results_df = pd.concat(results, ignore_index=True)

        results_df.to_csv(RESULTS_CSV_PATH, index=False)

        event("step", {"step": 2})

        cols = [
            "image_name", "final_danger_score", "mean_risk_score",
            "danger_level", "visual_risk_score",
            "predicted_weather", "weather_score",
            "num_people", "num_vehicles", "num_bikes",
        ]
        rows = []
        for _, row in results_df.iterrows():
            r = {}
            for c in cols:
                v = row.get(c, None)
                if v is None or (isinstance(v, float) and __import__("math").isnan(v)):
                    r[c] = None
                else:
                    r[c] = v
            rows.append(r)

        event("results", {"rows": rows})

        # ── Route recommendation ──────────────────────────────
        event("step", {"step": 3})
        print("\n================ ROUTE RECOMMENDATION ================")

        try:
            route_summary, route_decision, selected_route, notification = run_route_recommendation(
                RESULTS_CSV_PATH,
                min_risk_reduction_pct=min_risk_reduction_pct,
                max_extra_time_pct=max_extra_time_pct,
            )

            route_summary_path = OUTPUTS_DIR / "route_summary.csv"
            route_decision_path = OUTPUTS_DIR / "route_decision.csv"
            route_summary.to_csv(route_summary_path, index=False)
            route_decision.to_csv(route_decision_path, index=False)

            print(f"\nRecommended route: {selected_route['route_name']}")
            print(f"User notification: {notification}")

            summary_rows = []
            for _, r in route_summary.iterrows():
                summary_rows.append({
                    "route_name":       r.get("route_name"),
                    "avg_danger_score": r.get("avg_danger_score"),
                    "max_danger_score": r.get("max_danger_score"),
                    "travel_time_min":  r.get("travel_time_min"),
                    "num_images":       r.get("num_images"),
                    "total_people":     r.get("total_people"),
                    "total_vehicles":   r.get("total_vehicles"),
                    "danger_level":     r.get("danger_level"),
                })

            event("route", {
                "selected_route": selected_route["route_name"],
                "avg_danger":     float(selected_route["avg_danger_score"]),
                "travel_time":    int(selected_route["travel_time_min"]),
                "danger_level":   selected_route["danger_level"],
                "notification":   notification,
                "summary":        summary_rows,
            })

        except Exception as route_err:
            print(f"\nRoute recommendation failed: {route_err}")
            event("route_error", {"error": str(route_err)})

        print("\nPipeline complete.")
        event("done", {})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR:\n{tb}")
        event("error_fatal", {"error": str(e)})


@app.route("/api/stream/<run_id>")
def api_stream(run_id):
    q = _run_queues.get(run_id)
    if q is None:
        return "Not found", 404

    def generate():
        while True:
            try:
                item = q.get(timeout=60)
            except queue.Empty:
                yield "event: ping\ndata: {}\n\n"
                continue

            typ  = item.get("type")
            data = item.get("data", item)

            if typ == "log":
                payload = json.dumps({"text": item.get("text", "")})
                yield f"event: log\ndata: {payload}\n\n"
            elif typ in ("step", "results", "route", "route_error", "done", "error_fatal"):
                payload = json.dumps(data)
                yield f"event: {typ}\ndata: {payload}\n\n"

            if typ in ("done", "error_fatal"):
                _run_queues.pop(run_id, None)
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# Entry point
# ============================================================

def open_browser():
    import time
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("=" * 52)
    print("  NavSafe — Intelligent Route Safety Platform")
    print("=" * 52)
    print("\n  Opening dashboard at http://127.0.0.1:5000\n")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)