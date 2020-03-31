/* eslint-disable camelcase */
let host_url;
let drupal_host_url;
let ecpay_host_url;
let cas_host_url;
let domain_url;
let reward_domain_url;
let auto_login_url;
let paypal_client_id;
let asia_pay_show;
/* eslint-disable camelcase */
const root = "";

// development env for 'npm start'
if (process.env.NODE_ENV === "development") {
  host_url = process.env.REACT_APP_API_HOST_NAME_DEV;
  drupal_host_url = process.env.REACT_APP_DRUPAL_API_HOST_DEV;
  ecpay_host_url = process.env.REACT_APP_ECPAY_HOST_DEV;
  cas_host_url = process.env.REACT_APP_CAS_HOST_DEV;
  domain_url = process.env.REACT_APP_DOMAIN_DEV;
  reward_domain_url = process.env.REACT_APP_REWARD_DOMAIN_DEV;
  auto_login_url = process.env.REACT_APP_AUTO_LOGIN_URL_DEV;
  paypal_client_id = process.env.REACT_APP_PAYPAL_CLIENT_ID_DEV;
  asia_pay_show = process.env.REACT_APP_ASIA_PAY_SHOW_DEV;
  //testing env
} else if (process.env.REACT_APP_BUILD_ENV === "testing") {
  host_url = process.env.REACT_APP_API_HOST_NAME_TESTING;
  drupal_host_url = process.env.REACT_APP_DRUPAL_API_HOST_TESTING;
  ecpay_host_url = process.env.REACT_APP_ECPAY_HOST_TESTING;
  cas_host_url = process.env.REACT_APP_CAS_HOST_TESTING;
  domain_url = process.env.REACT_APP_DOMAIN_TESTING;
  reward_domain_url = process.env.REACT_APP_REWARD_DOMAIN_TESTING;
  auto_login_url = process.env.REACT_APP_AUTO_LOGIN_URL_TESTING;
  paypal_client_id = process.env.REACT_APP_PAYPAL_CLIENT_ID_TESTING;
  asia_pay_show = process.env.REACT_APP_ASIA_PAY_SHOW_TESTING;
  //uat env
} else if (process.env.REACT_APP_BUILD_ENV === "uat") {
  host_url = process.env.REACT_APP_API_HOST_NAME_UAT;
  drupal_host_url = process.env.REACT_APP_DRUPAL_API_HOST_UAT;
  ecpay_host_url = process.env.REACT_APP_ECPAY_HOST_UAT;
  cas_host_url = process.env.REACT_APP_CAS_HOST_UAT;
  domain_url = process.env.REACT_APP_DOMAIN_UAT;
  reward_domain_url = process.env.REACT_APP_REWARD_DOMAIN_UAT;
  auto_login_url = process.env.REACT_APP_AUTO_LOGIN_URL_UAT;
  paypal_client_id = process.env.REACT_APP_PAYPAL_CLIENT_ID_UAT;
  asia_pay_show = process.env.REACT_APP_ASIA_PAY_SHOW_UAT;
  //production env
} else if (process.env.REACT_APP_BUILD_ENV === "prod") {
  host_url = process.env.REACT_APP_API_HOST_NAME_PROD;
  drupal_host_url = process.env.REACT_APP_DRUPAL_API_HOST_PROD;
  ecpay_host_url = process.env.REACT_APP_ECPAY_HOST_PROD;
  cas_host_url = process.env.REACT_APP_CAS_HOST_PROD;
  domain_url = process.env.REACT_APP_DOMAIN_PROD;
  reward_domain_url = process.env.REACT_APP_REWARD_DOMAIN_PROD;
  auto_login_url = process.env.REACT_APP_AUTO_LOGIN_URL_PROD;
  paypal_client_id = process.env.REACT_APP_PAYPAL_CLIENT_ID_PROD;
  asia_pay_show = process.env.REACT_APP_ASIA_PAY_SHOW_PROD;
} else if (process.env.REACT_APP_BUILD_ENV === "ci") {
  host_url = process.env.REACT_APP_API_HOST_NAME_CI;
  drupal_host_url = process.env.REACT_APP_DRUPAL_API_HOST_CI;
  ecpay_host_url = process.env.REACT_APP_ECPAY_HOST_CI;
  cas_host_url = process.env.REACT_APP_CAS_HOST_CI;
  domain_url = process.env.REACT_APP_DOMAIN_CI;
  reward_domain_url = process.env.REACT_APP_REWARD_DOMAIN_CI;
  auto_login_url = process.env.REACT_APP_AUTO_LOGIN_URL_CI;
  paypal_client_id = process.env.REACT_APP_PAYPAL_CLIENT_ID_CI;
  asia_pay_show = process.env.REACT_APP_ASIA_PAY_SHOW_CI;
}

export const HOST_URL = host_url;
export const DRUPAL_HOST_URL = drupal_host_url;
export const ECPAY_HOST_URL = ecpay_host_url;
export const CAS_HOST_URL = cas_host_url;
export const DOMAIN_URL = domain_url;
export const REWARD_DOMAIN_URL = reward_domain_url;
export const AUTO_LOGIN_URL = auto_login_url;
export const ROOT = root;
export const PAYPAL_CLIENT_ID = paypal_client_id;
export const ASIA_PAY_SHOW = asia_pay_show;
