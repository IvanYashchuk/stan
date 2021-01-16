#include <stan/model/model_functional.hpp>
#include <test/unit/model/test_model_interface_rosenbrock.hpp>
#include <gtest/gtest.h>

TEST(ModelUtil, model_functionalModelBaseInterface) {
  int dim = 2;
  rosenbrock_model model(dim);

  EXPECT_THROW(stan::model::model_functional<rosenbrock_model>(model, 0),
               std::runtime_error);

  std::stringstream out;
  out.str("");
  EXPECT_THROW(stan::model::model_functional<rosenbrock_model>(model, &out),
               std::runtime_error);
  EXPECT_EQ("", out.str());
}
