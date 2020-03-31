import { combineReducers } from "redux";
import updateTheme from "./updateTheme";

const initialState = {
  active: true
};

const index = (state = initialState, action) => {
  switch (action.type) {
    case "SET_APP":
      return {
        ...state,
        active: false
      };

    default:
      return state;
  }
};

const reducers = combineReducers({
  index,
  updateTheme
});

export default reducers;
