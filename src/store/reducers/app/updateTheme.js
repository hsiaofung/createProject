const initialState = {
  getThemeSuccessfully: false,
  themeName: "primary"
};

export default (state = initialState, action) => {
  switch (action.type) {
    case "UPDATE_THEME":
      return {
        ...state,
        getThemeSuccessfully: true,
        themeName: action.themeName,
        themeColors: action.themeColors
      };

    default:
      return state;
  }
};
