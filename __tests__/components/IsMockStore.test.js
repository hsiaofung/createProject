import React from "react";
import Home from "../../src/pages/Home";
import { mount } from "enzyme";
import configureMockStore from "redux-mock-store";
import thunk from "redux-thunk";
import { Provider } from "react-redux";
import { mapDispatchToProps, mapStateToProps } from "../../src/pages/Home";

const middlewares = [thunk];
const mockStore = configureMockStore(middlewares);
const initalState = {
  app: {
    active: "123"
  }
};
const store = mockStore(initalState);

describe("", () => {
  const component = mount(
    <Provider store={store}>
      <Home />
    </Provider>
  );
  test("", () => {
    console.log(component.debug());
  });
});

describe("Dice", () => {
  it("should show previously rolled value", () => {
    const initialState = {
      app: {
        active: "123"
      }
    };
    expect(mapStateToProps(initialState)).toEqual({ active: "123" });
  });

  it("should roll the dice again when button is clicked", () => {
    const dispatch = jest.fn();
    mapDispatchToProps(dispatch).setApp();
    expect(dispatch.mock.calls[0]).toEqual([{ type: "SET_APP" }]);
  });
});
