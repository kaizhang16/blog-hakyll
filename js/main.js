Array.prototype.forEach.call(document.getElementsByTagName('table'), e => {
    e.classList.add('table');
    e.classList.add('table-striped');
    e.classList.add('table-bordered');
});

Array.prototype.forEach.call(document.getElementsByClassName('tags'), tags => {
    Array.prototype.forEach.call(tags.children, e => {
        e.classList.add('badge');
        e.classList.add('badge-light');
    });
});

const toc = document.getElementById('toc');
const newTOC = document.getElementById('newTOC');
if (toc) {
    newTOC.append(toc);
}
