import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";
import tw from "./locales/zhTW.json";

const resources = {
  en: {
    translation: en
  },
  zhTW: {
    translation: tw
  }
};

i18n.use(initReactI18next).init({
  resources,
  lng: "zhTW", //預設語言
  fallbackLng: "zhTW", //如果當前切換的語言沒有對應的翻譯則使用這個語言，
  interpolation: {
    escapeValue: false
  }
});
export default i18n;
