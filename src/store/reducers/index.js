import { combineReducers } from "redux";
// import * as ActionTypes from "../actionTypes";
import app from "./app";
import fetchTest from "./fetchTest";
import home from "./home";

const reducers = combineReducers({
  app,
  fetchTest,
  home
});

export default reducers;
