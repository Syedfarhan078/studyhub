# StudyHub — Cold-Start Pre-loader Page

A fully self-contained static HTML page that shows rotating educational flashcards while Render's free-tier spins up StudyHub (15-min inactivity cold start). It polls the backend every 3 seconds and auto-redirects the moment the server responds.

---

## 1. Set Your Render URL (Required!)

Open `index.html` and find line ~215:

```js
const STUDYHUB_URL = 'https://your-app.onrender.com';
```

Replace `your-app` with your actual Render subdomain, e.g.:

```js
const STUDYHUB_URL = 'https://studyhub-abc123.onrender.com';
```

---

## 2. Deploy Options

### Option A — Render Static Site (Recommended)
1. In your Render dashboard → **New → Static Site**
2. Connect the same GitHub repo
3. Set **Root Directory** → `wakeup-loader`
4. Set **Publish Directory** → `.` (a single dot)
5. Build Command → *(leave blank)*
6. Deploy → you get a free `*.onrender.com` URL for the loader itself

Then set **your main Render web service's Custom Domain** (or just share the loader URL) so users land on the loader first.

### Option B — GitHub Pages
1. Push your repo to GitHub
2. Settings → Pages → Source: `main` branch, folder `/wakeup-loader`
3. GitHub gives you `https://yourusername.github.io/studyhub/`

### Option C — Netlify Drop
Drag the `wakeup-loader/` folder onto [netlify.com/drop](https://app.netlify.com/drop) — instant deploy, free HTTPS URL.

---

## 3. Optional — Prevent Cold Starts Entirely (Free Render Cron)

Render's free tier spins down after 15 minutes of inactivity.
You can prevent this with a **free Render Cron Job** that pings your app every 13 minutes:

1. Render dashboard → **New → Cron Job**
2. Command: `curl https://your-app.onrender.com/?wake=1`
3. Schedule: `*/13 * * * *`
4. Instance type: **Free**

This keeps your Render service permanently warm — the pre-loader page becomes a fallback only for the first deploy or manual restarts.

---

## 4. What the Page Does

| Feature | Detail |
|---|---|
| **Design** | Matches StudyHub's warm paper aesthetic (Source Serif 4 + Source Sans 3, navy/paper tokens) |
| **Logo** | Embedded as base64 — no external asset requests |
| **Progress bar** | Fills over ~55 s with realistic stage labels |
| **Flashcards** | 15 rotating cards: Study Methods, Exam Tips, Quick Concepts, Did You Know? |
| **Polling** | `fetch` with `mode: 'no-cors'` every 3 s — detects opaque success vs network error |
| **Auto-redirect** | 1.4 s after server responds — smooth UX |
| **Self-contained** | Single HTML file, zero build step, deployable anywhere |
