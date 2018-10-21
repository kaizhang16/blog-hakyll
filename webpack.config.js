const path = require('path');

module.exports = {
    entry: './src/elm-mdc.js',
    output: {
        filename: 'elm-mdc.js',
        path: path.resolve(__dirname, '_site')
    }
};
