import { setApp } from "./index";

export const fetchData = () => dispatch => {
  // 一樣可以去呼叫其他不在這隻檔案的事件
  dispatch(setApp());
  // 或是這隻檔案的事件
  dispatch(fetchApple());
};

// return dispatch 用法
export function fetchTesting() {
  return function(dispatch) {
    // 一樣可以去呼叫其他不在這隻檔案的事件
    dispatch(setApp());
    // 或是這隻檔案的事件
    dispatch(fetchApple());
  };
}

export const fetchApple = () => {
  // 記得使用了return就不能再加上dispatch
  return {
    type: "FETCH_APPLE"
  };
};
