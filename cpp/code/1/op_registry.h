#include <map>
#include <iostream>

class opregistry{
public:
    static opregistry& instance() {
        static opregistry ops;
        return ops;
    }

    void insert(std::string op_name) {
        auto res = ops_map.insert(std::pair<std::string, std::string>(op_name, op_name));
        if(res.second == false)
            std::cout << op_name << " is already exists" << std::endl;
        else {
            std::cout << op_name << " register success" << std::endl;
            ops_map[op_name] = op_name;
        }
    }

    std::string creatOp(std::string op_name) {
        auto res = ops_map.find(op_name);
        if(res == ops_map.end())
            std::cout << op_name << " is don't exists" << std::endl;
        else
            return ops_map[op_name];
    }

    opregistry(const opregistry&) = delete;
    opregistry(opregistry&&) = delete;
private:
    std::map<std::string, std::string>ops_map;
    opregistry() = default;
};

struct regisrar{
    regisrar(std::string op_name) {
        std::cout << "regisrar" << std::endl;
        opregistry::instance().insert(op_name);
    }
};
