#pragma once

#define CHECK_TYPE(x, type) (std::type_index(typeid(type)) == std::type_index(typeid(x)))