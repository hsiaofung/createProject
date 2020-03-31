import getCookie from "../../function/getCookie";
import writeCookie from "../../function/writeCookie";

// 基本型
export const setApp = () => {
  return {
    type: "SET_APP"
  };
};

// 呼叫其他 action 型
export const selectLv1Option = index => async dispatch => {
  await dispatch(setApp(index));
};

// 執行多件 function 型
export const fetchProduct = cbu => async dispatch => {
  fetch(`/shopping/v1/compositions/productDetails/info/${cbu}`, {
    credentials: "include",
    headers: { "content-type": "application/json" },
    mode: "cors"
  })
    .then(response => response.json())
    .then(data => {
      console.log(data);
    });
};

const primaryTheme = {
  background: "#FFFFFF",
  textColor: "#000000",
  button: "red"
};

const darkTheme = {
  background: "#2F2F2F",
  textColor: "#CCCCCC",
  button: "darkred"
};

export const readThemeCookie = () => dispatch => {
  // 更換網站介面顏色
  const theme_cookie = getCookie("theme");
  if (!theme_cookie) {
    // 1. 預設主題顏色為亮色
    writeCookie("theme", "primary");
    const themeName = "primary";
    const themeColors = primaryTheme;
    dispatch(updateTheme(themeName, themeColors));
  } else {
    // 1. 抓取目前的主題顏色
    const themeName = getCookie("theme");
    const themeColors = themeName === "primary" ? primaryTheme : darkTheme;
    dispatch(updateTheme(themeName, themeColors));
  }
};

export const changeTheme = themeName => dispatch => {
  if (themeName === "primary") {
    const themeColors = primaryTheme;
    writeCookie("theme", themeName);
    dispatch(updateTheme(themeName, themeColors));
  } else {
    const themeColors = darkTheme;
    writeCookie("theme", themeName);
    dispatch(updateTheme(themeName, themeColors));
  }
};

export const updateTheme = (themeName, themeColors) => {
  return {
    type: "UPDATE_THEME",
    themeName,
    themeColors
  };
};

// TEST
export const addCounter = () => {
  return {
    type: "ADD_COUNTER",
    payload: { addQuantity: 1 }
  };
};

// Fetch TEST
export const fetchCount = () => async dispatch => {
  try {
    const response = await fetch("http://example.com/count");
    const result = await response.json();
    console.log("result:", result);
    dispatch(updateCount(result));
  } catch (error) {
    console.log("error:", error);
  } finally {
  }
};

export const updateCount = result => ({
  type: "UPDATE_COUNTER",
  result
});
