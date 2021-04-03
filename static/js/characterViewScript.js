let imgUploaded = false;
let uploadedImg = null;

document.addEventListener("DOMContentLoaded", function () {
    let infoBtn = document.getElementById("info");
    infoBtn.style.display = "none";

    suggest();

    imgUploaded = false;
    const inputImg = document.getElementById("inputimg");
    const canvas = document.getElementById("canvas");
    inputImg.style.display = "none";
    canvas.style.display = "block";

    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function draw(e) {
        if (!isDrawing) return;

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    canvas.addEventListener("mousedown", (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", () => isDrawing = false);
    canvas.addEventListener("mouseout", () => isDrawing = false);

    let fileTag = document.getElementById("filetag");
    let preview = document.getElementById("preview");

    fileTag.addEventListener("change", function () {
        changeImage(this);
    });

    function changeImage(input) {
        imgUploaded = true;
        let reader;

        if (input.files && input.files[0]) {
            inputImg.style.display = "block";
            canvas.style.display = "none";
            reader = new FileReader();

            reader.onload = function (e) {
                preview.setAttribute('src', e.target.result);
                uploadedImg = e.target.result;
            }

            reader.readAsDataURL(input.files[0]);
        }
    }
});

function submitImage() {
    let img = null;
    if (!imgUploaded) {
        let canvas = document.getElementById("canvas");
        img = canvas.toDataURL();
    } else {
        img = uploadedImg;
    }

    console.log("SUBMIT");
    predictImage(img);
    suggest();
}

function predictImage(img) {
    fetch("/character/predict/", {
        method: "POST",
        body: img
    }).then((data) => {
        if (data.ok) {
            return data.text();
        } else {
            throw Error(data.statusText);
        }
    }).then((data) => {
        console.log(data);
        let reponseObj = JSON.parse(data);

        let firstGuessClass = document.getElementById("firstGuessClass");
        firstGuessClass.setAttribute("value", reponseObj._firstGuessClass);

        let guess = document.getElementById("guess");
        guess.textContent = reponseObj._firstGuess;

        let confidence = document.getElementById("confidence");
        confidence.textContent = "(confidence: " + reponseObj._firstGuessConfidentLvl + "%)";

        let guess2Heading = document.getElementById("guess2Heading");
        guess2Heading.textContent = "Second guess :";

        let guess2 = document.getElementById("guess2");
        guess2.textContent = reponseObj._secondGuess;

        let confidence2 = document.getElementById("confidence2");
        confidence2.textContent = "(confidence: " + reponseObj._secondGuessConfidentLvl + "%)";

        let infoBtn = document.getElementById("info");
        infoBtn.style.display = "block";
    }).catch((error) => {
        console.log(error);
    });
}

function clearCanvas() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (imgUploaded) {
        let inputImg = document.getElementById("inputimg");
        let canvas = document.getElementById("canvas");
        let fileTag = document.getElementById("filetag");
        let preview = document.getElementById("preview");
        inputImg.style.display = "none";
        canvas.style.display = "block";
        fileTag.setAttribute('value', "");
        preview.setAttribute('src', "");
        imgUploaded = false;
    }

    const guess = document.getElementById("guess");
    guess.textContent = "";

    const confidence = document.getElementById("confidence");
    confidence.textContent = "";

    const guess2Heading = document.getElementById("guess2Heading");
    guess2Heading.textContent = "";

    const guess2 = document.getElementById("guess2");
    guess2.textContent = "";

    const confidence2 = document.getElementById("confidence2");
    confidence2.textContent = "";

    let infoBtn = document.getElementById("info");
    infoBtn.style.display = "none";
}

function suggest() {
    const suggestion = document.getElementById("suggestion");
    fetch("/character/suggest/").then(response => response.text())
        .then(data => suggestion.textContent = "Not sure what to draw??  Try " + data + " .");
}

function openModal() {
    let characterClass = document.getElementById("firstGuessClass").value;
    fetch("/character/info/" + characterClass, {
        method: "GET"
    }).then((data) => {
        if (data.ok) {
            return data.text();
        } else {
            throw Error(data.statusText);
        }
    }).then((data) => {
        console.log(data);
        let responseObj = JSON.parse(data);
        document.getElementById("characterName").textContent = responseObj._name;
        if (characterClass !== "15") {
            document.getElementById("character").textContent = responseObj._character;
            document.getElementById("character").style.display = "block";
            document.getElementById("characterKo").style.display = "none";
        } else {
            document.getElementById("characterKo").textContent = responseObj._character;
            document.getElementById("characterKo").style.display = "block";
            document.getElementById("character").style.display = "none";
        }

        document.getElementById("unicode").textContent = ": U + " + responseObj._unicode;
        document.getElementById("phonetic").textContent = ": " + responseObj._phonetic;
        document.getElementById("group").textContent = ": " + responseObj._group;
        document.getElementById("description").textContent = ": " + responseObj._description;

        let audioSrcResponse = responseObj._audio;
        let splittedStr = audioSrcResponse.split("'");
        let audioSrc = splittedStr[1];
        document.getElementById("audioSource").src = "data:audio/mp3;base64," + audioSrc;
        document.getElementById("audio").load();

        let modal = document.getElementById("myModal");
        modal.style.display = "block";
    }).catch((error) => {
        console.log(error);
    });
}

function closeModal(){
    let modal = document.getElementById("myModal");
    modal.style.display = "none";
}

window.onclick = function (event) {
    let modal = document.getElementById("myModal");

    if (event.target == modal) {
        modal.style.display = "none";
    }
}