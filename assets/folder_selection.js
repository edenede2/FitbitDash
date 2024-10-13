document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.folder-upload').forEach(function (input) {
        input.addEventListener('click', function (event) {
            event.target.querySelector('input').setAttribute('webkitdirectory', 'true');
            event.target.querySelector('input').setAttribute('directory', 'true');
        });
    });
});
