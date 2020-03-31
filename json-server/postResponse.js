module.exports = function(req, res, next) {
  if (req.method === "POST" && req.url === "/addresses") {
    res.method = "GET";
    res.sendStatus(200);
    console.log("Address............", req.url);
  }
  if (req.method === "GET" && req.url === "/login") {
    res.sendStatus(201);
    if (res.statusCode === 201) console.log("Signed In............", req.url);
    if (res.statusCode === 200) console.log("Not Login ............", req.url);
  }
  if (req.method === "POST" && req.url === "/myFav") {
    res.sendStatus(200);
    console.log(
      "Change POST to GET while sending myFav API............",
      req.url
    );
  }
  if (req.method === "POST" && req.url === "/tempOrder") {
    // Force to send status equal 200
    res.sendStatus(200);
    res.send({
      errors: [
        {
          message: "The inventory quantity is not enough.",
          errorCode: "E00949"
        }
      ]
    });
    console.log(
      "Change POST to GET while sending temp Order API............",
      req.url
    );
  }
  // Continue to JSON Server router
  next();
};
