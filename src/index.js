import React from "react";
import ReactDOM from "react-dom";
import "./index.css";
import App from "./App";
import * as serviceWorker from "./serviceWorker";
import { Provider } from "react-redux";
import store from "./store";
// 載入i18n語言包設定
import "./i18n";
// 解決 IE 瀏覽器問題的套件
import "babel-polyfill";
import "isomorphic-fetch";
import "react-app-polyfill/ie11";
import "react-app-polyfill/stable";
import "core-js/features/promise";

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById("main")
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
