import configureMockStore from "redux-mock-store";
import thunk from "redux-thunk";
import fetchMock from "fetch-mock";
import { addCounter, fetchCount, updateCount } from "../../src/store/actions";
import { fetchTesting } from "../../src/store/actions/home";
import reducer from "../../src/store/reducers/fetchTest";

const middlewares = [thunk];
const mockStore = configureMockStore(middlewares);

describe("addCount", () => {
  // 每一次測試後清除 fetchMock 的紀錄
  afterEach(() => {
    fetchMock.restore();
  });

  test("test actions", () => {
    const expectAction = {
      type: "ADD_COUNTER",
      payload: { addQuantity: 1 }
    };
    expect(addCounter()).toEqual(expectAction);
  });

  test("get count dispatch of action", () => {
    // 創建 store
    const store = mockStore({ count: 0 });
    const expectedActions = [{ type: "SET_APP" }, { type: "FETCH_APPLE" }];
    // 1. 執行fetchTesting()
    store.dispatch(fetchTesting());
    // 2. 是否執行SET_APP及FETCH_APPLE
    expect(store.getActions()).toEqual(expectedActions);
  });

  test("get count dispatch of action", () => {
    // fetchMock 與 fetchCount() 內的請求網址相同
    fetchMock.getOnce("http://example.com/count", {
      body: { count: 3 }
    });

    // 創建 store
    const store = mockStore({ count: 0 });
    const expectedActions = [{ type: "UPDATE_COUNTER", result: { count: 3 } }];

    // 使用 store 用 fetchCount() 觸發 dispatch
    return store.dispatch(fetchCount()).then(() => {
      // 這裡可以看到 dispatch 觸發了哪些事件
      expect(store.getActions()).toEqual(expectedActions);
    });
  });

  test("test reducer", () => {
    // 確認初始資料
    const initialData = { count: 0, request: false };
    expect(reducer(undefined, {})).toEqual(initialData);

    // 傳入初始值及 addCounter ：
    // 確認回傳的 object count 是否正確 + 1
    expect(reducer(initialData, addCounter())).toEqual({
      count: 1,
      request: true
    });
    expect(reducer(initialData, updateCount({ count: 2 }))).toEqual({
      count: { count: 2 },
      request: false
    });
  });
});
