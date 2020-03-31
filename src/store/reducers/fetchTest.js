const initState = {
  count: 0,
  request: false
};

const reducer = (state = initState, action) => {
  switch (action.type) {
    case "ADD_COUNTER":
      return {
        ...state,
        count: state.count + 1,
        request: true
      };
    case "UPDATE_COUNTER":
      return {
        ...state,
        count: action.result
      };
    default:
      return state;
  }
};

export default reducer;
