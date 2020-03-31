const path = require("path");
const jsonServer = require("json-server");
const server = jsonServer.create();
const router = jsonServer.router(path.join(__dirname, "db.json"));
const middlewares = jsonServer.defaults();
const postResponse = require(path.join(__dirname, "postResponse.js"));
const routes = require(path.join(__dirname, "routes.json"));

// 模擬 POST 這支API時 resoponse 寫進回傳資料
server.post("/cm/v1/campaigns", function(req, res) {
  // http://localhost:3000 是網站的Domain網域 (也有可能是8080)
  res.header("Access-Control-Allow-Origin", "http://localhost:3000");
  res.header("Access-Control-Allow-Credentials", "true");
  // 回傳的資料為{ region: "TWN", campaignCode: "MR" }
  res.jsonp({ region: "TWN", campaignCode: "MR" });
});

server.use(middlewares);
server.use(jsonServer.rewriter(routes));
server.use(postResponse);
server.use(router);
server.listen(8081, () => {
  console.log("JSON Server is running");
});
