const body = document.body.style;
const bodyWidth = document.body.clientWidth;

export const lockBody = () => {
  body.overflow = "hidden";
  if (bodyWidth > 1024) {
    body.paddingRight = "17px";
  }
};

export const unlockBody = () => {
  body.overflow = "";
  body.paddingRight = "";
};
