function updateCurrentDateTime() {
    const target = document.getElementById("currentDateTime");
    if (!target) return;

    const formatter = new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short"
    });
    target.textContent = formatter.format(new Date());
}

function applyStoredPreferences() {
    const html = document.documentElement;
    const body = document.body;
    const storedTheme = localStorage.getItem("anemiavision-theme");
    const largeText = localStorage.getItem("anemiavision-large-text") === "true";

    if (storedTheme === "dark") {
        html.setAttribute("data-theme", "dark");
    }
    if (largeText) {
        body.classList.add("large-text");
    }
}

function bindPreferences() {
    const html = document.documentElement;
    const body = document.body;
    const darkModeToggle = document.getElementById("darkModeToggle");
    const largeTextToggle = document.getElementById("largeTextToggle");

    if (darkModeToggle) {
        darkModeToggle.setAttribute("aria-pressed", String(html.getAttribute("data-theme") === "dark"));
        darkModeToggle.addEventListener("click", () => {
            const isDark = html.getAttribute("data-theme") === "dark";
            html.setAttribute("data-theme", isDark ? "light" : "dark");
            localStorage.setItem("anemiavision-theme", isDark ? "light" : "dark");
            darkModeToggle.setAttribute("aria-pressed", String(!isDark));
        });
    }

    if (largeTextToggle) {
        largeTextToggle.setAttribute("aria-pressed", String(body.classList.contains("large-text")));
        largeTextToggle.addEventListener("click", () => {
            body.classList.toggle("large-text");
            const enabled = body.classList.contains("large-text");
            localStorage.setItem("anemiavision-large-text", String(enabled));
            largeTextToggle.setAttribute("aria-pressed", String(enabled));
        });
    }
}

function bindSidebar() {
    const sidebar = document.getElementById("sidebar");
    const openButton = document.querySelector("[data-sidebar-open]");
    const closeButton = document.querySelector("[data-sidebar-close]");

    if (!sidebar) return;
    if (openButton) {
        openButton.addEventListener("click", () => sidebar.classList.add("open"));
    }
    if (closeButton) {
        closeButton.addEventListener("click", () => sidebar.classList.remove("open"));
    }
}

function animateConfidenceMeters() {
    document.querySelectorAll(".confidence-fill[data-progress]").forEach((bar) => {
        const progress = Number(bar.dataset.progress || 0);
        requestAnimationFrame(() => {
            bar.style.width = `${Math.max(0, Math.min(100, progress))}%`;
        });
    });
}

function bindToasts() {
    if (!window.bootstrap) return;
    document.querySelectorAll(".toast").forEach((toastElement) => {
        const toast = new window.bootstrap.Toast(toastElement);
        toast.show();
    });
}

function bindKeyboardShortcuts() {
    document.addEventListener("keydown", (event) => {
        const target = event.target;
        const tagName = target?.tagName?.toLowerCase();
        const isTypingField =
            tagName === "input" ||
            tagName === "textarea" ||
            tagName === "select" ||
            Boolean(target?.isContentEditable);

        if (isTypingField || event.altKey || event.ctrlKey || event.metaKey) {
            return;
        }

        if (event.key === "n" || event.key === "N") {
            window.location.href = "/scan";
        }
        if (event.key === "h" || event.key === "H") {
            window.location.href = "/history";
        }
    });
}

function bindPrintButtons() {
    ["printReportButton", "printReportButtonSecondary"].forEach((id) => {
        const button = document.getElementById(id);
        if (!button) return;
        button.addEventListener("click", () => window.print());
    });
}

function revealSkeletons() {
    window.setTimeout(() => {
        document.querySelectorAll("[data-chart-shell]").forEach((shell) => {
            shell.classList.remove("is-loading");
        });
    }, 1200);
}

document.addEventListener("DOMContentLoaded", () => {
    applyStoredPreferences();
    updateCurrentDateTime();
    setInterval(updateCurrentDateTime, 1000 * 30);
    bindPreferences();
    bindSidebar();
    animateConfidenceMeters();
    bindToasts();
    bindKeyboardShortcuts();
    bindPrintButtons();
    revealSkeletons();

    if (window.lucide) {
        window.lucide.createIcons();
    }
});
