let squareCounter = 1;

document.getElementById("add-square").addEventListener("click", function () {
    const widthInput = document.querySelector(".width").value;
    const heightInput = document.querySelector(".height").value;

    if (widthInput && heightInput) {
        const width = parseInt(widthInput, 10);
        const height = parseInt(heightInput, 10);

        if (width > 20 || height > 20) {
            alert("Error: Width and Height must be 20 cm or less.");
            return;
        }

        const squareList = document.getElementById("square-list");
        const listItem = document.createElement("li");
        listItem.textContent = `Square ${squareCounter}: Width ${width} x Height ${height}`;
        squareList.appendChild(listItem);

        const squareCount = document.getElementById("square-count");
        squareCount.textContent = parseInt(squareCount.textContent) + 1;
        squareCounter++;

        document.querySelector(".width").value = "";
        document.querySelector(".height").value = "";
    } else {
        alert("Please enter both width and height.");
    }
});

document.getElementById("calculate").addEventListener("click", function () {
    const squares = getSquaresFromInputs();
    fetch("/calculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ squares: squares })
    })
    .then(response => response.blob())
    .then(imageBlob => {
        const imgElement = document.getElementById("result-image");
        imgElement.src = URL.createObjectURL(imageBlob);
    })
    .catch(error => console.error("Error:", error));
});

function getSquaresFromInputs() {
    const squares = [];
    document.querySelectorAll("#square-list li").forEach(listItem => {
        const [width, height] = listItem.textContent.match(/\d+/g).slice(1).map(Number);
        squares.push({ width, height });
    });
    return squares;
}
