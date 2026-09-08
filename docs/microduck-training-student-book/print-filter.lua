function Image(image)
  if FORMAT:match("latex") then
    for _, class in ipairs(image.classes) do
      if class == "book-cover" then
        return {}
      end
    end
    image.src = image.src:gsub("%.svg$", ".pdf")
    image.attributes.height = "7.6in"
    if image.src:match("studio%-.*%-summary") then
      image.attributes.width = "4.7in"
    elseif image.src:match("studio%-parameters") then
      image.attributes.width = "4.7in"
    elseif image.src:match("studio%-evaluation") then
      image.attributes.width = "4.4in"
    else
      image.attributes.width = "6.65in"
    end
  end
  return image
end

function Code(code)
  if FORMAT:match("latex") and #code.text > 28 and code.text:match("^[%w%._/%-]+$") then
    local pieces = {}
    local count = 0
    for character in code.text:gmatch(".") do
      table.insert(pieces, character == "_" and "\\_" or character)
      count = count + 1
      if character:match("[_/%-]") or count == 8 then
        table.insert(pieces, "\\allowbreak{}")
        count = 0
      end
    end
    return pandoc.RawInline("latex", "\\texttt{" .. table.concat(pieces) .. "}")
  end
  return code
end

function Header(header)
  if FORMAT:match("latex") and header.identifier == "the-students-exact-61-observations" then
    return {pandoc.RawBlock("latex", "\\clearpage"), header}
  end
  return header
end

function Figure(figure)
  if FORMAT:match("latex") and pandoc.utils.stringify(figure.caption.long):match("Four accepted simulated scenarios") then
    return {}
  end
  return figure
end
