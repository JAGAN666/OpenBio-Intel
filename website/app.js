(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ---------- Figure 1: question resolves into a cited answer ---------- */
  const answer = document.getElementById("answer");
  const refs = document.querySelector(".refs-live");
  const q = document.getElementById("queryText");
  const steps = [...document.querySelectorAll("#trace span")];
  const claims = [...document.querySelectorAll("#response .claim")];
  const notes = [...document.querySelectorAll("#sidenotes li")];
  const noteFor = (n) => notes.find((li) => li.dataset.ref === n);
  let run = 0;

  function showAll() {
    answer.classList.remove("pending");
    answer.classList.add("complete");
    refs.classList.remove("pending");
    q.textContent = q.dataset.full;
    q.classList.remove("typing");
    claims.forEach((c) => c.classList.add("shown", "lit"));
  }

  async function play() {
    const id = ++run;
    const alive = () => id === run;
    answer.classList.add("pending");
    answer.classList.remove("complete");
    refs.classList.add("pending");
    claims.forEach((c) => c.classList.remove("shown", "lit"));
    notes.forEach((n) => n.classList.remove("shown", "flash"));
    steps.forEach((s) => s.classList.remove("on", "done"));

    const text = q.dataset.full;
    q.textContent = "";
    q.classList.add("typing");
    await wait(350);
    for (let i = 1; i <= text.length; i++) {
      if (!alive()) return;
      q.textContent = text.slice(0, i);
      await wait(text[i - 1] === " " ? 38 : 22);
    }
    q.classList.remove("typing");

    for (const s of steps) {
      if (!alive()) return;
      s.classList.add("on");
      await wait(520);
      s.classList.add("done");
    }
    await wait(200);
    if (!alive()) return;
    answer.classList.add("complete");

    for (const c of claims) {
      if (!alive()) return;
      c.classList.add("shown");
      await wait(260);
      c.classList.add("lit");
      const n = noteFor(c.dataset.ref);
      if (n) {
        n.classList.add("shown", "flash");
        setTimeout(() => n.classList.remove("flash"), 900);
      }
      await wait(520);
    }
    answer.classList.remove("pending");
    refs.classList.remove("pending");
  }

  // Hovering or focusing a claim points at its source in the margin.
  claims.forEach((c) => {
    const on = () => noteFor(c.dataset.ref)?.classList.add("flash");
    const off = () => noteFor(c.dataset.ref)?.classList.remove("flash");
    c.addEventListener("mouseenter", on);
    c.addEventListener("mouseleave", off);
    c.addEventListener("focusin", on);
    c.addEventListener("focusout", off);
  });

  document.getElementById("replay").addEventListener("click", () => (reduce ? showAll() : play()));

  if (reduce) showAll();
  else play();

  /* ---------- Figure 6: bars grow when scrolled into view ---------- */
  const bars = document.querySelector(".bars");
  if (bars && !reduce && "IntersectionObserver" in window) {
    bars.classList.add("animate");
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        bars.classList.add("in");
        io.disconnect();
      }
    }, { threshold: 0.4 });
    io.observe(bars);
  }

  /* ---------- copy buttons ---------- */
  const toast = document.getElementById("toast");
  let toastTimer;
  function say(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
  }

  document.querySelectorAll(".copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = document.getElementById(btn.dataset.copy).innerText.trim();
      try {
        await navigator.clipboard.writeText(text);
        btn.textContent = "Copied";
        say("Copied to clipboard");
      } catch {
        const range = document.createRange();
        range.selectNodeContents(document.getElementById(btn.dataset.copy));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        say("Selected. Press Ctrl+C or ⌘C to copy");
      }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  });
})();
