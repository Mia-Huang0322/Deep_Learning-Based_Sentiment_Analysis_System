document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const submitButton = form.querySelector("button");

    form.addEventListener("submit", function (e) {
        submitButton.disabled = true; // 禁用提交按钮，防止多次提交
        submitButton.textContent = "Analyzing..."; // 改变按钮文字
    });
});
