// Shared UI behavior: sidebar collapse, modals, pill-tabs, toggles.
(function () {
  const sidebar = document.getElementById("sidebar");
  const toggleBtn = document.getElementById("sidebarToggle");
  if (toggleBtn && sidebar) {
    const KEY = "iv_sidebar_collapsed";
    if (localStorage.getItem(KEY) === "1") {
      sidebar.classList.add("collapsed");
      toggleBtn.textContent = "›";
    }
    toggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
      const collapsed = sidebar.classList.contains("collapsed");
      toggleBtn.textContent = collapsed ? "›" : "‹";
      localStorage.setItem(KEY, collapsed ? "1" : "0");
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
    group.querySelectorAll("[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        group.querySelectorAll("[data-tab]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const val = btn.getAttribute("data-tab");
        document.querySelectorAll(`[data-tabpanel="${groupName}"]`).forEach((panel) => {
          panel.style.display = panel.getAttribute("data-tabvalue") === val ? "" : "none";
        });
      });
    });
  });

  // Toggle switches: [data-toggle] flips .on class (visual only unless data-toggle-name present -> posts to endpoint)
  document.querySelectorAll(".toggle[data-toggle]").forEach((t) => {
    t.addEventListener("click", () => t.classList.toggle("on"));
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
