var icon=document.getElementById("icon");
icon.onclick=function(){
    document.body.classList.toggle("dark-theme")
    if(document.body.classList.contains("dark-theme")){
    icon.src="brightness.png"
    }else{
        icon.src="night-mode.png";
    }
}

var dropdown = document.querySelector(".dropdown");
var dropdownContent = document.querySelector(".dropdown-content");

// Toggle dropdown visibility on button click
dropdown.addEventListener("click", function() {
    dropdownContent.classList.toggle("show");
});

// Close the dropdown if clicked outside of it
window.addEventListener("click", function(event) {
    if (!dropdown.contains(event.target)) {
        dropdownContent.classList.remove("show");
    }
});




