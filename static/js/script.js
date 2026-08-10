function copyCoverLetter() {

    const text =
        document.getElementById("coverLetter").innerText;

    navigator.clipboard.writeText(text);

    alert("Cover Letter Copied!");
}