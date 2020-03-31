const initState = {
  fetched: true
};

const reducer = (state = initState, action) => {
  switch (action.type) {
    case "FETCH_DATA":
      return {
        ...state,
        fetched: false
      };
    case "FETCH_APPLE":
      return {
        ...state,
        fetched: "apple"
      };
    default:
      return state;
  }
};

export default reducer;
