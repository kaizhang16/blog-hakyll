Array.prototype.forEach.call(document.getElementsByTagName('table'), t => {
    t.classList.add('table');
    t.classList.add('table-striped');
    t.classList.add('table-bordered');
});
