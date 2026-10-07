document.addEventListener("DOMContentLoaded", () => {
    const search = document.getElementById("globalSearch");
    if (search) {
        search.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && search.value.trim()) {
                window.location.href = "/analysis?search=" + encodeURIComponent(search.value.trim());
            }
        });
    }
});
