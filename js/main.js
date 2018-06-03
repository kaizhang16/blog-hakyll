const toc = document.getElementById('toc');
if (toc) {
    const newTOC = document.getElementById('newTOC');
    const tocTitle = document.createElement('h1');
    tocTitle.setAttribute('id', 'tocTitle');
    tocTitle.appendChild(document.createTextNode("目录"));
    newTOC.append(tocTitle);
    newTOC.append(toc);
    newTOC.style.display = 'block';
}

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
