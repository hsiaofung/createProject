import { mapDispatchToProps, mapStateToProps } from "../../src/pages/Home";

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
    mapDispatchToProps(dispatch).fetchTesting();
    expect(dispatch.mock.calls[0][0]).toEqual({ type: "SET_APP" });
  });
});
