// Shared UI behavior: sidebar collapse, modals, pill-tabs.
(function () {
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.getElementById("sidebarToggle");
  if (toggleBtn && sidebar) {
    const KEY = "iv_sidebar_collapsed";

    // The arrow direction is handled in CSS (.sidebar.collapsed rotates the
    // chevron), so this only has to manage state and the accessible labels.
    function apply(collapsed) {
      sidebar.classList.toggle("collapsed", collapsed);
      const label = collapsed ? "Expand sidebar" : "Collapse sidebar";
      toggleBtn.setAttribute("title", label);
      toggleBtn.setAttribute("aria-label", label);
      toggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }

    apply(localStorage.getItem(KEY) === "1");

    toggleBtn.addEventListener("click", () => {
      const collapsed = !sidebar.classList.contains("collapsed");
      apply(collapsed);
      localStorage.setItem(KEY, collapsed ? "1" : "0");
      hideSidebarTooltip();
    });
  }

  // Mobile nav: below 860px the sidebar is a slide-in drawer (see .sidebar in
  // style.css), opened by the topbar's hamburger button and closed by its own
  // backdrop. Independent of the desktop collapse toggle above - "collapsed"
  // (icons-only) and "mobile-open" (full drawer over content) are different
  // states that never apply at the same time, since the breakpoints don't
  // overlap.
  const menuToggle = document.getElementById("menuToggle");
  const backdrop = document.getElementById("sidebarBackdrop");
  if (menuToggle && sidebar && backdrop) {
    function setMobileOpen(open) {
      sidebar.classList.toggle("mobile-open", open);
      backdrop.classList.toggle("show", open);
      menuToggle.setAttribute("aria-expanded", open ? "true" : "false");
      menuToggle.setAttribute("aria-label", open ? "Close navigation menu" : "Open navigation menu");
      document.body.classList.toggle("no-scroll", open);
    }
    menuToggle.addEventListener("click", () => setMobileOpen(!sidebar.classList.contains("mobile-open")));
    backdrop.addEventListener("click", () => setMobileOpen(false));
  }

  // Collapsed-sidebar tooltips: a nav icon alone doesn't say what it does, and
  // hovering the rail used to show nothing. A floating label fixes that
  // without the whole-page reflow a hover-to-expand sidebar would cause.
  // position:fixed (not absolute) is deliberate - .sidebar-nav scrolls its own
  // content and clips anything absolutely positioned past its right edge, so
  // only a viewport-fixed element can sit outside the rail.
  let sidebarTip = null;
  function hideSidebarTooltip() {
    if (sidebarTip) sidebarTip.classList.remove("show");
  }
  if (sidebar) {
    sidebar.querySelectorAll(".sidebar-nav .nav-item").forEach((item) => {
      item.addEventListener("mouseenter", () => {
        if (!sidebar.classList.contains("collapsed")) return;
        const label = item.querySelector(".label");
        if (!label) return;
        if (!sidebarTip) {
          sidebarTip = document.createElement("div");
          sidebarTip.className = "sidebar-tooltip";
          document.body.appendChild(sidebarTip);
        }
        sidebarTip.textContent = label.textContent;
        const r = item.getBoundingClientRect();
        sidebarTip.style.top = r.top + r.height / 2 + "px";
        sidebarTip.style.left = r.right + 10 + "px";
        sidebarTip.classList.add("show");
      });
      item.addEventListener("mouseleave", hideSidebarTooltip);
    });
  }

  // Modals: any [data-open-modal="modalId"] opens #modalId; any [data-close-modal] closes nearest .modal-scrim
  document.querySelectorAll("[data-open-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-open-modal");
      const modal = document.getElementById(id);
      if (modal) modal.classList.add("open");
    });
  });
  document.querySelectorAll("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const scrim = btn.closest(".modal-scrim");
      if (scrim) scrim.classList.remove("open");
    });
  });
  document.querySelectorAll(".modal-scrim").forEach((scrim) => {
    scrim.addEventListener("click", (e) => {
      if (e.target === scrim) scrim.classList.remove("open");
    });
  });

  // Pill tabs: [data-tabgroup] wraps [data-tab] buttons; [data-tabpanel] panels toggle by matching value
  document.querySelectorAll("[data-tabgroup]").forEach((group) => {
    const groupName = group.getAttribute("data-tabgroup");
    function activate(val) {
      const btn = group.querySelector(`[data-tab="${val}"]`);
      if (!btn) return false;
      group.querySelectorAll("[data-tab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(`[data-tabpanel="${groupName}"]`).forEach((panel) => {
        panel.style.display = panel.getAttribute("data-tabvalue") === val ? "" : "none";
      });
      return true;
    }

    group.querySelectorAll("[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => activate(btn.getAttribute("data-tab")));
    });

    // Open a specific tab from the URL, e.g. /rhu/risk#atrisk. Lets a link
    // point at the tab a user actually wants rather than the page default.
    if (window.location.hash) {
      const val = window.location.hash.slice(1);
      if (activate(val)) {
        // The panel was display:none when the page loaded, so the browser could
        // not scroll to it natively. Scroll once it is visible.
        const panel = document.querySelector(
          `[data-tabpanel="${groupName}"][data-tabvalue="${val}"]`);
        if (panel) {
          requestAnimationFrame(() =>
            panel.scrollIntoView({ behavior: "smooth", block: "start" }));
        }
      }
    }
  });

  // Generic confirm-on-click for destructive actions
  document.querySelectorAll("[data-confirm]").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (!confirm(el.getAttribute("data-confirm"))) {
        e.preventDefault();
        e.stopPropagation();
      }
    });
  });
})();

/* ---------------------------------------------------------------------------
   Account menu (topbar). Closes on outside click, on Escape, and returns focus
   to the trigger so keyboard users are not stranded inside a closed menu.
--------------------------------------------------------------------------- */
(function () {
  var btn = document.getElementById("accountBtn");
  var menu = document.getElementById("accountDropdown");
  if (!btn || !menu) return;

  function open() {
    menu.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    var first = menu.querySelector("[role=menuitem]");
    if (first) first.focus();
  }
  function close(refocus) {
    menu.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    if (refocus) btn.focus();
  }

  btn.addEventListener("click", function (e) {
    e.stopPropagation();
    if (menu.hidden) { open(); } else { close(false); }
  });

  document.addEventListener("click", function (e) {
    if (!menu.hidden && !menu.contains(e.target) && e.target !== btn) close(false);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !menu.hidden) close(true);
  });

  // Arrow keys move between items once the menu is open.
  menu.addEventListener("keydown", function (e) {
    var items = Array.prototype.slice.call(menu.querySelectorAll("[role=menuitem]"));
    var i = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  });
})();

/* ---------------------------------------------------------------------------
   Child Records: collapse the 15 dose columns.
   They sit between the identity columns and Actions, so with them shown the
   action buttons are off screen and need a sideways scroll. Collapsed by
   default; the choice is remembered.
--------------------------------------------------------------------------- */
(function () {
  var btn = document.getElementById("doseToggle");
  if (!btn) return;
  var label = document.getElementById("doseToggleLabel");
  var table = btn.closest("table");
  var KEY = "iv_doses_expanded";

  function apply(expanded) {
    table.classList.toggle("doses-expanded", expanded);
    btn.setAttribute("aria-expanded", expanded ? "true" : "false");
    label.textContent = expanded ? "Hide doses" : "Doses";
    btn.title = expanded ? "Hide the individual dose columns"
                         : "Show the individual dose columns";
  }

  apply(localStorage.getItem(KEY) === "1");

  btn.addEventListener("click", function () {
    var expanded = !table.classList.contains("doses-expanded");
    apply(expanded);
    localStorage.setItem(KEY, expanded ? "1" : "0");
  });
})();

/* ---------------------------------------------------------------------------
   Custom select: wraps every select.form-control in a styled trigger+panel
   (see .select-panel in style.css) so the open dropdown matches the rest of
   the app instead of the browser's unstyleable native option list. The real
   <select> stays in the DOM, visually hidden, as the actual form field - a
   change here sets its value and dispatches a real 'change' event, so every
   onchange="..." and addEventListener('change', ...) a template already has
   keeps working unmodified. If this script fails to load, the select is
   simply left as a normal (still chevron-fixed, see style.css) native one.
--------------------------------------------------------------------------- */
(function () {
  var CHEVRON_SVG = '<svg class="st-chevron" width="13" height="13" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="6 9 12 15 18 9"></polyline></svg>';
  var uid = 0;

  function closeAllPanels(except) {
    document.querySelectorAll(".select-panel").forEach(function (panel) {
      if (panel === except || panel.hidden) return;
      panel.hidden = true;
      var trigger = panel.parentElement.querySelector(".select-trigger");
      if (trigger) trigger.setAttribute("aria-expanded", "false");
    });
  }

  function enhance(select) {
    if (select.dataset.customized) return;
    select.dataset.customized = "1";
    uid += 1;

    var shell = document.createElement("div");
    shell.className = "select-shell";
    if (select.getAttribute("style")) shell.setAttribute("style", select.getAttribute("style"));
    select.parentNode.insertBefore(shell, select);
    shell.appendChild(select);

    var trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = select.className + " select-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    if (select.disabled) trigger.disabled = true;
    if (select.id) trigger.setAttribute("aria-labelledby", select.id);

    var valueSpan = document.createElement("span");
    valueSpan.className = "st-value";
    trigger.appendChild(valueSpan);
    trigger.insertAdjacentHTML("beforeend", CHEVRON_SVG);
    shell.appendChild(trigger);

    var panel = document.createElement("div");
    panel.className = "select-panel";
    panel.setAttribute("role", "listbox");
    panel.hidden = true;
    shell.appendChild(panel);

    var options = [];
    function buildOptions() {
      panel.innerHTML = "";
      options = Array.prototype.map.call(select.options, function (opt, i) {
        var row = document.createElement("div");
        row.className = "select-option";
        row.id = "select-opt-" + uid + "-" + i;
        row.setAttribute("role", "option");
        row.title = opt.textContent;
        var label = document.createElement("span");
        label.className = "so-label";
        label.textContent = opt.textContent;
        row.appendChild(label);
        if (opt.disabled) row.setAttribute("aria-disabled", "true");
        row.dataset.value = opt.value;
        panel.appendChild(row);
        return row;
      });
      syncFromSelect();
    }

    function syncFromSelect() {
      var selectedOpt = select.options[select.selectedIndex];
      valueSpan.textContent = selectedOpt ? selectedOpt.textContent : "";
      options.forEach(function (row) {
        row.classList.toggle("is-selected", row.dataset.value === select.value);
      });
    }

    function setActive(row) {
      options.forEach(function (r) { r.classList.remove("is-active"); });
      if (!row) return;
      row.classList.add("is-active");
      trigger.setAttribute("aria-activedescendant", row.id);
      row.scrollIntoView({ block: "nearest" });
    }

    function openPanel() {
      closeAllPanels(panel);
      panel.hidden = false;
      trigger.setAttribute("aria-expanded", "true");
      setActive(panel.querySelector(".is-selected") || options[0]);
    }
    function closePanel(refocus) {
      panel.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      trigger.removeAttribute("aria-activedescendant");
      if (refocus) trigger.focus();
    }
    function choose(row) {
      if (!row || row.getAttribute("aria-disabled") === "true") return;
      select.value = row.dataset.value;
      syncFromSelect();
      select.dispatchEvent(new Event("change", { bubbles: true }));
      closePanel(true);
    }

    trigger.addEventListener("click", function () {
      if (panel.hidden) openPanel(); else closePanel(false);
    });
    trigger.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (panel.hidden) { openPanel(); return; }
        var i = options.indexOf(panel.querySelector(".is-active"));
        var next = e.key === "ArrowDown" ? Math.min(i + 1, options.length - 1) : Math.max(i - 1, 0);
        setActive(options[next]);
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (panel.hidden) openPanel(); else choose(panel.querySelector(".is-active"));
      } else if (e.key === "Escape" && !panel.hidden) {
        e.preventDefault();
        closePanel(true);
      } else if (e.key === "Tab") {
        closePanel(false);
      }
    });
    panel.addEventListener("mousemove", function (e) {
      var row = e.target.closest(".select-option");
      if (row) setActive(row);
    });
    panel.addEventListener("click", function (e) {
      var row = e.target.closest(".select-option");
      if (row) choose(row);
    });

    // Keeps the trigger's label correct if something else (e.g. a reset
    // button, or a filter script elsewhere on the page) changes the
    // underlying select's value directly.
    select.addEventListener("change", syncFromSelect);

    buildOptions();
  }

  document.querySelectorAll("select.form-control").forEach(enhance);

  document.addEventListener("click", function (e) {
    if (!e.target.closest(".select-shell")) closeAllPanels(null);
  });
})();
