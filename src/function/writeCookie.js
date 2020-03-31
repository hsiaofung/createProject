//name : type string
const writeCookie = (name, data) => {
  document.cookie = name.toString() + '=' + data + ';path=/';
  return;
};

export default writeCookie;
